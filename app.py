# app.py

from flask import Flask, render_template, request, Response, jsonify, send_file
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import cvzone
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
import yt_dlp
import os
import re # Added for robust URL parsing

# Import modules
import config 
import data_analysis
from yolov8.tracker import Tracker # assuming tracker is available

# ------------------------------
# Flask App Setup
# ------------------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ------------------------------
# Global State for Streaming
# ------------------------------
capture_queues = {}
process_out_queues = {}
stream_threads = {}
stop_events = {}
predicted_output = 0 # current prediction level for map color

# ------------------------------
# Helpers
# ------------------------------

def get_livestream_url(youtube_url: str) -> str:
    """Uses yt_dlp to get the direct stream URL."""
    ydl_opts = {
        'cookiefile': config.cookies_file,
        'format': 'best',
        'quiet': True,
        'skip_download': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            return info_dict['url']
    except Exception as e:
        print(f"Error fetching direct stream URL for {youtube_url}: {e}")
        return None


# ------------------------------
# Threading/Streaming Logic
# ------------------------------

def start_capture_thread(stream_url: str, frame_queue: Queue, stop_event):
    """Captures frames from the stream and handles reconnection attempts."""
    cap = cv2.VideoCapture(stream_url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    while not stop_event.is_set():
        ret, frame = cap.read()
        
        if not ret:
            # Reconnection Logic (Fix for stuck video)
            print("Capture failed. Re-opening stream...")
            cap.release()
            time.sleep(1.0) # Wait a moment before retrying the connection
            
            # Re-initialize the video capture object
            cap = cv2.VideoCapture(stream_url) 
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            continue # Skip the rest of the loop and try reading a frame again

        frame = cv2.resize(frame, (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT))

        try:
            frame_queue.put_nowait(frame)
        except:
            # Aggressively discard old frame if queue is full to ensure low latency
            try:
                _ = frame_queue.get_nowait()
            except Empty:
                pass
            try:
                frame_queue.put_nowait(frame)
            except:
                pass

    cap.release()

def start_processing_thread(video_index: int, frame_queue: Queue, out_frame_queue: Queue, data_queue: Queue, stop_event):
    # Call data_analysis.save_counts_row
    # Use config constants (model_yolo, PRED_INTERVAL_SECONDS, etc.)
    tracker = Tracker()
    global predicted_output
    
    with open("yolov8/coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    wait_between = 1.0 / config.PROCESS_FPS

    while not stop_event.is_set():
        start = time.time()
        try:
            frame = frame_queue.get(timeout=1.0)
        except Empty:
            continue

        # run inference
        # 🎯 FIX 2: Aggressively reduced YOLO resolution to imgsz=320 for MAX speed
        results = config.model_yolo.predict(frame, imgsz=320, conf=0.35, verbose=False) 
        detections = results[0].boxes.data
        px = pd.DataFrame(detections).astype("float") if detections is not None else pd.DataFrame()

        cars, buses, trucks = [], [], []
        for _, row in px.iterrows():
            x1, y1, x2, y2, conf, cls_id = int(row[0]), int(row[1]), int(row[2]), int(row[3]), row[4], int(row[5])
            c = class_list[cls_id].lower()
            if 'car' in c:
                cars.append([x1, y1, x2, y2])
            elif 'bus' in c:
                buses.append([x1, y1, x2, y2]) 
            elif 'truck' in c:
                trucks.append([x1, y1, x2, y2])

        cars_boxes = tracker.update(cars)
        buses_boxes = tracker.update(buses)
        trucks_boxes = tracker.update(trucks)

        # Draw boxes & labels
        draw_frame = frame.copy()
        for bbox in cars_boxes:
            cv2.rectangle(draw_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
            cvzone.putTextRect(draw_frame, 'Car', (bbox[0], bbox[1] - 10), 1, 1)
        for bbox in buses_boxes:
            cv2.rectangle(draw_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
            cvzone.putTextRect(draw_frame, 'Bus', (bbox[0], bbox[1] - 10), 1, 1)
        for bbox in trucks_boxes:
            cv2.rectangle(draw_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
            cvzone.putTextRect(draw_frame, 'Truck', (bbox[0], bbox[1] - 10), 1, 1)

        car_count = len(cars_boxes)
        bus_count = len(buses_boxes)
        truck_count = len(trucks_boxes)
        total_count = car_count + bus_count + truck_count

        # Non-blocking data push to the data saving thread (This logic is correct)
        try:
            data_queue.put_nowait([car_count, bus_count, truck_count, total_count])
        except:
            pass 

        cvzone.putTextRect(draw_frame,
                            f'Car: {car_count}, Bus: {bus_count}, Truck: {truck_count}, Total: {total_count}',
                            (10, config.CAPTURE_HEIGHT - 10), 1, 1)

       # encode and put into out queue
        ret, buffer = cv2.imencode('.jpg', draw_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            
            # 🎯 FIX 3: Aggressive Non-Blocking Output Logic (Flush queue before putting)
            try:
                # Clear all old frames to make sure the newest frame is displayed immediately
                while True:
                    out_frame_queue.get_nowait()
            except Empty:
                pass
            
            try:
                out_frame_queue.put_nowait(frame_bytes)
            except:
                # Should only fail if something major is wrong
                pass

        # enforce processing fps cap
        elapsed = time.time() - start
        if elapsed < wait_between:
            time.sleep(wait_between - elapsed)

def start_data_saving_thread(video_index: int, data_queue: Queue, stop_event):
    """Saves averaged vehicle counts to CSV on a fixed interval (non-blocking to video)."""
    last_save = time.time()
    
    while not stop_event.is_set():
        time.sleep(0.5) # Check every half-second
        
        all_counts = []
        try:
            while True:
                all_counts.append(data_queue.get_nowait()) 
        except Empty:
            pass
            
        now = time.time()
        
        if now - last_save >= config.PRED_INTERVAL_SECONDS and all_counts:
            avg = np.mean(all_counts, axis=0)
            
            data_analysis.save_counts_row(
                ts=now, video_index=video_index,
                car=int(round(avg[0])), bus=int(round(avg[1])),
                truck=int(round(avg[2])), total=int(round(avg[3]))
            )
            last_save = now

def generate_frames_from_queue(out_frame_queue: Queue, stop_event):
    """Generator function to stream JPEG frames from the output queue to the web browser."""
    while not stop_event.is_set():
        try:
            frame_bytes = out_frame_queue.get(timeout=1.0)
        except Empty:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# app.py

def ensure_stream_started(video_index: int):
    # Uses config constants
    if video_index in stream_threads:
        return

    url = get_livestream_url(config.youtube_urls[video_index])

    frame_q = Queue(maxsize=config.QUEUE_MAXSIZE)
    # FIX: Output queue remains 10 for smoothness
    out_q = Queue(maxsize=10) 
    # 🎯 FIX 1: New queue for non-blocking data (Increased size)
    data_q = Queue(maxsize=500) 
    
    # ... (rest of the function remains the same) ...
    
    stop_ev = threading.Event()

    t_cap = threading.Thread(target=start_capture_thread, args=(url, frame_q, stop_ev), daemon=True)
    t_proc = threading.Thread(target=start_processing_thread, args=(video_index, frame_q, out_q, data_q, stop_ev), daemon=True)
    t_data = threading.Thread(target=start_data_saving_thread, args=(video_index, data_q, stop_ev), daemon=True)

    t_cap.start()
    t_proc.start()
    t_data.start()

    capture_queues[video_index] = frame_q
    process_out_queues[video_index] = out_q
    stream_threads[video_index] = (t_cap, t_proc, t_data)
    stop_events[video_index] = stop_ev


# ------------------------------
# Flask routes
# ------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    global predicted_output
    video_index = request.args.get('video_index', default=0, type=int)
    city = config.CITY_MAP.get(video_index, f"City_{video_index}") 

    # --- NEW LOGIC FOR YOUTUBE EMBED URL ---
    youtube_url = config.youtube_urls[video_index]
    # Use regex to find the video ID for robust extraction
    match = re.search(r'(?<=v=)[\w-]+|(?<=youtu\.be/)[\w-]+', youtube_url)
    video_id = match.group(0) if match else None
    # app.py -> index() function

    youtube_embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1" if video_id else ""
    # ----------------------------------------

    geojson_data = config.load_geojson()
    df = pd.DataFrame({'name': ['Heavy', 'High', 'Low', 'Normal'], 'value': [0, 1, 2, 3]})

    centers = [
        {"lat": 35.6936, "lon": 139.6991, "name": 'Shinjukugado-W, Tokyo, Japan'},
        {"lat": 39.5473, "lon": -107.3247, "name": 'Colorado Mountain College, USA'},
        {"lat": 40.7579, "lon": -73.9855, "name": 'Times Square, New York, USA'},
        {"lat": 33.3973, "lon": -104.5227, "name": 'Roswell, New Mexico, USA'},
    ]
    c = centers[min(max(video_index, 0), len(centers) - 1)]

    # Plotly Map Setup (unchanged)
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_data,
        locations='name',
        color='name',
        mapbox_style='open-street-map',
        zoom=15,
        center={"lat": c["lat"], "lon": c["lon"]},
        opacity=0.5,
        labels={'name': 'Traffic Situation'},
        width=750,
        height=560,
        color_discrete_map={"Heavy": "red", "High": "green", "Low": "blue", 'Normal': 'yellow'},
    )

    line_data = pd.DataFrame({'lat': [c['lat'] + 0.0006, c['lat'] - 0.0006], 'lon': [c['lon'] - 0.0003, c['lon'] + 0.0003]})
    fig.add_trace(go.Scattermapbox(
        lat=line_data['lat'],
        lon=line_data['lon'],
        mode='lines+markers',
        marker=go.scattermapbox.Marker(size=8),
        line=dict(width=4, color='blue'),
        name=c['name'],
    ))

    line_color = config.COLOR_MAP.get(int(predicted_output), 'gray')
    fig.data[-1].line.color = line_color
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    graph_html = fig.to_html(full_html=False)

    # Call the 24-hour prediction function
    predictions_24h = data_analysis.predict_next_24_hours(city) 
    preds = predictions_24h.get('predictions') if predictions_24h.get('ok') else []

    # Calculate total and level based on the full 24-hour prediction
    if preds:
        total_24h = sum(p['predicted_total'] for p in preds)
        
        # Adjusting thresholds for a full 24-hour total
        if total_24h < 5000:
            total_level = "Low"
        elif total_24h < 15000:
            total_level = "Medium"
        elif total_24h < 30000:
            total_level = "High"
        else:
            total_level = "Heavy"
    else:
        total_24h = None
        total_level = None

    return render_template(
        'index.html',
        graph_html=graph_html,
        video_index=video_index,
        youtube_embed_url=youtube_embed_url, # NEW: Pass the embed URL
        hourly_preds=preds, 
        total_24h=total_24h, 
        total_level=total_level,
        city=city
    )

@app.route('/video_feed')
def video_feed():
    video_index = request.args.get('video_index', default=0, type=int)
    ensure_stream_started(video_index)
    out_q = process_out_queues.get(video_index)
    if out_q is None:
        return "Stream not available", 503

    return Response(generate_frames_from_queue(out_q, stop_events[video_index]),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict_tomorrow')
def predict_tomorrow_api():
    video_index = request.args.get("video_index", default=0, type=int)
    city = config.CITY_MAP.get(video_index, f"City_{video_index}")
    out = data_analysis.predict_next_24_hours(city)
    return jsonify(out)

@app.route('/data.csv')
def download_csv():
    data_analysis._ensure_counts_csv()
    return send_file(
        config.CSV_FILE,
        as_attachment=True,
        download_name="traffic_counts.csv",
        mimetype='text/csv'
    )

# ------------------------------
# App Startup
# ------------------------------
if __name__ == '__main__':
    # Start all streams on startup
    for video_index in config.CITY_MAP.keys():
        ensure_stream_started(video_index)
    app.run(debug=True, threaded=True)
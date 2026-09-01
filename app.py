import eventlet
eventlet.monkey_patch()
import os
import cv2
import numpy as np
import base64
import threading
import time
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from ultralytics import YOLO
from collections import deque
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__,
            static_folder=os.path.join(basedir, 'static'),
            template_folder=os.path.join(basedir, 'templates'))
app.config['SECRET_KEY'] = 'your-secret-key-here-make-it-very-secure'
app.config['UPLOAD_FOLDER'] = 'uploads'

CORS(app)

socketio = SocketIO(app,
                    async_mode='eventlet',
                    cors_allowed_origins="*",
                    logger=True,
                    engineio_logger=True,
                    max_http_buffer_size=100 * 1024 * 1024)  # 100MB


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'MD Najiur Rana' and password == 'Rana2026':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Incorrect username or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

try:

    drone_model_path = os.path.join(basedir, "static", "models", "drone_detection.pt")
    if os.path.exists(drone_model_path):
        model = YOLO(drone_model_path)
        logger.info("The drone-specific model was successfully loaded.")
    else:
        model = YOLO("yolov8n.pt")
        logger.info("Standard YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

class SimpleTracker:
    def __init__(self, max_distance=50, max_missing=5):
        self.next_id = 0
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missing = max_missing
    def update(self, detections):

        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['missing'] += 1
                if self.tracks[track_id]['missing'] > self.max_missing:
                    del self.tracks[track_id]
            return self.tracks

        if len(self.tracks) > 0:
            track_ids = list(self.tracks.keys())
            track_points = np.array([self.tracks[track_id]['position'] for track_id in track_ids])
            detection_points = np.array([det['center'] for det in detections])

            distances = np.linalg.norm(track_points[:, np.newaxis] - detection_points, axis=2)

            matched_detections = set()
            matched_tracks = set()

            while np.min(distances) < self.max_distance:

                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                track_idx, det_idx = min_idx

                track_id = track_ids[track_idx]
                self.tracks[track_id]['position'] = detections[det_idx]['center']
                self.tracks[track_id]['missing'] = 0
                self.tracks[track_id]['class'] = detections[det_idx]['class']

                matched_detections.add(det_idx)
                matched_tracks.add(track_idx)

                distances[track_idx, :] = float('inf')
                distances[:, det_idx] = float('inf')
                if np.min(distances) >= self.max_distance:
                    break

            for i, det in enumerate(detections):
                if i not in matched_detections:
                    self.tracks[self.next_id] = {
                        'position': det['center'],
                        'missing': 0,
                        'class': det['class'],
                        'path': deque(maxlen=30)
                    }
                    self.tracks[self.next_id]['path'].append(det['center'])
                    self.next_id += 1

            for i, track_id in enumerate(track_ids):
                if i not in matched_tracks:
                    self.tracks[track_id]['missing'] += 1
                    if self.tracks[track_id]['missing'] > self.max_missing:
                        del self.tracks[track_id]
        else:

            for det in detections:
                self.tracks[self.next_id] = {
                    'position': det['center'],
                    'missing': 0,
                    'class': det['class'],
                    'path': deque(maxlen=30)
                }
                self.tracks[self.next_id]['path'].append(det['center'])
                self.next_id += 1

        for track_id in self.tracks:
            if self.tracks[track_id]['missing'] == 0:
                self.tracks[track_id]['path'].append(self.tracks[track_id]['position'])
        return self.tracks

tracker = SimpleTracker()
processing = False
current_video_path = None

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))
@socketio.on('connect')
def handle_connect():
    logger.info('Client connection successful')
    emit('connection_status', {'status': 'connected', 'message': 'Connected to server'})
@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnects')
@socketio.on('client_connected')
def handle_client_connected(data):
    logger.info(f'Client connected: {data}')
    emit('connection_established', {'message': 'Connection established, processing can begin.'})
@socketio.on('heartbeat')
def handle_heartbeat(data):
    emit('heartbeat_response', {'timestamp': data['timestamp'], 'server_time': time.time()})
@socketio.on('upload_image')
def handle_image_upload(data):
    try:
        if model is None:
            emit('processing_error', {'message': 'The model was not loaded correctly.'})
            return

        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img, verbose=False)
        annotated_img = results[0].plot()

        cls_counts = {}
        detected_objects = []
        for box in results[0].boxes:
            if box.cls is not None and box.conf is not None:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = results[0].names[cls_id]

                if conf > 0.3:
                    cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1
                    detected_objects.append({
                        'class': cls_name,
                        'confidence': conf,
                        'class_id': cls_id
                    })

        logger.info(f"Detected objects: {detected_objects}")
        logger.info(f"Statistical results: {cls_counts}")

        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        emit('image_processed', {
            'processed_image': f'data:image/jpeg;base64,{img_str}',
            'stats': cls_counts,
            'message': f"Image analysis complete! Detected. {sum(cls_counts.values())} One object"
        })
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        emit('processing_error', {'message': f'Error processing image: {str(e)}'})
@socketio.on('upload_video')
def handle_video_upload(data):
    global processing, current_video_path
    try:
        logger.info("Received video upload request")
        if model is None:
            emit('processing_error', {'message': 'The model was not loaded correctly.'})
            return

        if processing:
            emit('processing_error', {'message': 'The video is currently being processed; please wait for it to complete.'})
            return

        filename = data['filename']
        logger.info(f"Processing video files: {filename}")

        if 'video' not in data:
            emit('processing_error', {'message': 'Video data missing'})
            return

        video_data_str = data['video']
        if ',' in video_data_str:
            video_data_str = video_data_str.split(',')[1]
        video_data = base64.b64decode(video_data_str)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(video_path, 'wb') as f:
            f.write(video_data)
        current_video_path = video_path
        processing = True

        thread = threading.Thread(
            target=process_video,
            args=(video_path, data.get('enable_tracking', False))
        )
        thread.daemon = True
        thread.start()
        emit('video_processing_started', {
            'message': 'Start processing video',
            'filename': filename
        })
    except Exception as e:
        logger.error(f"Error processing video upload: {e}")
        emit('processing_error', {'message': f'Error while processing video: {str(e)}'})
@socketio.on('stop_processing')
def handle_stop_processing():
    global processing
    processing = False
    logger.info("Received a stop processing request")
def process_video(video_path, enable_tracking):
    global processing
    try:
        logger.info(f"Start processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f'Unable to open video file: {video_path}'
            logger.error(error_msg)
            socketio.emit('processing_error', {'message': error_msg})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = 0
        cls_counts = {}
        logger.info(f"Video Information - Total Frames: {total_frames}, FPS: {fps}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_width = 640
        if width > 0:
            scale_factor = target_width / width
        else:
            scale_factor = 1.0
        while processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("Video loading finished or an error occurred.")
                break
            try:

                if width > target_width:
                    frame = cv2.resize(frame, (target_width, int(height * scale_factor)))

                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()

                current_detections = []
                for box in results[0].boxes:
                    if box.cls is not None and box.conf is not None:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = results[0].names[cls_id]

                        if conf > 0.3:
                            cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1

                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            current_detections.append({
                                'class': cls_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y]
                            })

                if enable_tracking:
                    tracks = tracker.update(current_detections)
                    for track_id, track_info in tracks.items():
                        path = list(track_info['path'])
                        if len(path) > 1:

                            if track_info['class'] == 'car':
                                color = (0, 0, 255)
                            elif track_info['class'] == 'truck':
                                color = (255, 0, 0)
                            elif track_info['class'] == 'bus':
                                color = (0, 165, 255)
                            elif track_info['class'] == 'motorcycle':
                                color = (0, 255, 0)
                            else:
                                color = (0, 255, 0)

                            for i in range(1, len(path)):
                                cv2.line(annotated_frame,
                                         (int(path[i - 1][0]), int(path[i - 1][1])),
                                         (int(path[i][0]), int(path[i][1])),
                                          color, 3)

                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_str = base64.b64encode(buffer).decode('utf-8')

                socketio.emit('video_frame_processed', {
                    'processed_frame': f'data:image/jpeg;base64,{frame_str}',
                    'stats': cls_counts,
                    'progress': int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                })
                frame_count += 1

                delay = 1.0 / fps if fps > 0 else 0.03
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue
        cap.release()
        if processing:
            completion_msg = f"Video analysis complete! Detected {sum(cls_counts.values())} One object"
            logger.info(completion_msg)
            socketio.emit('processing_complete', {
                'message': completion_msg,
                'stats': cls_counts
            })
    except Exception as e:
        error_msg = f'Error while processing video: {str(e)}'
        logger.error(error_msg)
        socketio.emit('processing_error', {'message': error_msg})
    finally:
        processing = False
        logger.info("Video processing completed")
if __name__ == '__main__':
    logger.info("Server startup")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False)
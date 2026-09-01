import sys
import os
import cv2
import numpy as np
import json
from collections import deque
from ultralytics import YOLO
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QGraphicsView, QTextEdit,
                               QFileDialog, QMessageBox, QProgressBar, QGraphicsScene,
                               QGroupBox, QFrame, QSpacerItem, QSizePolicy, QDialog,
                               QLineEdit, QFormLayout, QDialogButtonBox)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QPixmap, QImage
class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log in to the system")
        self.setFixedSize(400, 200)
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #1e3c72;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a5298;
            }
        """)
        self.init_ui()
        self.username = None
        self.password = None
    def init_ui(self):
        layout = QVBoxLayout(self)

        title_label = QLabel("UAV Recognition System Based on YOLO")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e3c72; margin-bottom: 20px;")
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Please enter your username")
        form_layout.addRow("Username:", self.username_edit)
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Please enter your password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Password:", self.password_edit)
        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    def accept(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()

        if username == "MD Najiur Rana" and password == "Rana2026":
            self.username = username
            self.password = password
            super().accept()
        else:
            QMessageBox.warning(self, "Login Failed", "Incorrect username or password")
            self.password_edit.clear()

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
                    'path': deque(maxlen=30)  # 存储轨迹点
                }
                self.tracks[self.next_id]['path'].append(det['center'])
                self.next_id += 1

        for track_id in self.tracks:
            if self.tracks[track_id]['missing'] == 0:  # 仅更新当前帧中存在的轨迹
                self.tracks[track_id]['path'].append(self.tracks[track_id]['position'])
        return self.tracks
class VideoProcessor(QObject):
    frame_processed = Signal(np.ndarray, np.ndarray, dict)
    progress_updated = Signal(int)
    analysis_finished = Signal(str)
    stats_updated = Signal(dict)
    def __init__(self, model, video_path, enable_tracking=False):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.is_running = False
        self.tracker = SimpleTracker()
        self.enable_tracking = enable_tracking
        self.cls_counts = {}
    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.analysis_finished.emit(f"Unable to open video file: {self.video_path}")
                return
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            self.is_running = True
            self.cls_counts = {}
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, verbose=False)
                annotated_frame = results[0].plot()  # 这个会自动绘制边界框

                current_detections = []
                for box in results[0].boxes:
                    if box.cls is not None and box.conf is not None:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = results[0].names[cls_id]

                        if conf > 0.5:

                            self.cls_counts[cls_name] = self.cls_counts.get(cls_name, 0) + 1

                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2

                            current_detections.append({
                                'class': cls_name,
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y]
                            })

                if self.enable_tracking:
                    tracks = self.tracker.update(current_detections)

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

                self.frame_processed.emit(frame, annotated_frame, self.cls_counts)
                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)

                cv2.waitKey(int(1000 / fps))
            cap.release()
            self.stats_updated.emit(self.cls_counts)
            self.analysis_finished.emit(f"Video analysis complete! Detected {sum(self.cls_counts.values())} objects")
        except Exception as e:
            self.analysis_finished.emit(f"Error processing video: {str(e)}")
    def stop_processing(self):
        self.is_running = False
class ImageProcessor(QObject):
    image_processed = Signal(np.ndarray, np.ndarray, dict)
    analysis_finished = Signal(str)
    def __init__(self, model):
        super().__init__()
        self.model = model
    def process_image(self, image_path):
        try:

            image = cv2.imread(image_path)
            if image is None:
                self.analysis_finished.emit(f"Unable to read image: {image_path}")
                return

            results = self.model(image, verbose=False)
            annotated_image = results[0].plot()

            cls_counts = {}
            for box in results[0].boxes:
                if box.cls is not None and box.conf is not None:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = results[0].names[cls_id]

                    if conf > 0.5:
                        cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1

            self.image_processed.emit(image, annotated_image, cls_counts)
            self.analysis_finished.emit(f"Image analysis complete! Detected {sum(cls_counts.values())} objects")
        except Exception as e:
            self.analysis_finished.emit(f"Error processing image: {str(e)}")
class MainWindow(QMainWindow):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.setWindowTitle(f"UAV Recognition System Based on YOLO - user: {username}")
        self.resize(1600, 900)

        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #1e3c72, stop: 1 #2a5298);
            }
            QWidget {
                background: transparent;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 5px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #2c3e50;
                font-weight: bold;
                background-color: white;
            }
            QPushButton {
                background-color: #1e3c72;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #2a5298;
            }
            QPushButton:pressed {
                background-color: #162b4d;
            }
            QPushButton#detectButton {
                background-color: #1e90ff;
            }
            QPushButton#analyzeButton {
                background-color: #ff7f50;
            }
            QLabel {
                padding: 5px;
                color: white;
                font-size: 14px;
            }
            QLabel#titleLabel {
                font-size: 24px;
                font-weight: bold;
                color: white;
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 15px;
            }
            QTextEdit {
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 8px;
                background-color: white;
                color: #2c3e50;
                font-size: 12px;
            }
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                background-color: white;
                color: #2c3e50;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #1e90ff;
                width: 10px;
                border-radius: 3px;
            }
            QComboBox {
                padding: 8px;
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: white;
                font-size: 14px;
                min-height: 20px;
            }
            QFrame#imageFrame, QFrame#videoFrame {
                background-color: white;
                border-radius: 8px;
                border: 2px solid #1e3c72;
            }
            QLabel#sectionLabel {
                background-color: #1e3c72;
                color: white;
                font-weight: bold;
                font-size: 16px;
                padding: 10px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QLabel#groupTitle {
                font-weight: bold;
                color: #2c3e50;
                background-color: white;
                padding: 5px;
                border-bottom: 2px solid #cccccc;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
        """)

        try:

            drone_model_path = "models/drone_detection.pt"
            if os.path.exists(drone_model_path):
                self.model = YOLO(drone_model_path)
                print("The drone-specific model was loaded successfully")
            else:
                self.model = YOLO("yolov8n.pt")
                print("The standard YOLO model is loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model = None
            QMessageBox.critical(self, "错误", f"Model loading failed: {e}")

        self.current_video = None
        self.current_image = None
        self.video_processor = None
        self.image_processor = None
        self.video_thread = None
        self.image_thread = None
        self.init_ui()
    def init_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        control_panel = QWidget()
        control_panel.setFixedWidth(350)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(5, 5, 5, 5)

        title_label = QLabel(f"UAV Recognition System Based on YOLO\nuser: {self.username}")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        control_layout.addWidget(title_label)

        control_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        image_group = QGroupBox("")
        image_layout = QVBoxLayout(image_group)
        image_layout.setContentsMargins(0, 0, 0, 10)

        image_title = QLabel("Image Processing")
        image_title.setObjectName("groupTitle")
        image_title.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(image_title)
        self.image_label = QLabel("No image selected")
        self.image_label.setStyleSheet("color: #2c3e50; background-color: #f0f0f0; padding: 8px; border-radius: 4px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.image_label)
        select_image_btn = QPushButton("Select images")
        select_image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(select_image_btn)
        analyze_image_btn = QPushButton("Detection images")
        analyze_image_btn.setObjectName("detectButton")
        analyze_image_btn.clicked.connect(self.analyze_image)
        image_layout.addWidget(analyze_image_btn)
        control_layout.addWidget(image_group)

        video_group = QGroupBox("")
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(0, 0, 0, 10)

        video_title = QLabel("Video Processing")
        video_title.setObjectName("groupTitle")
        video_title.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(video_title)
        self.video_label = QLabel("No video selected")
        self.video_label.setStyleSheet("color: #2c3e50; background-color: #f0f0f0; padding: 8px; border-radius: 4px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        select_video_btn = QPushButton("Select Video")
        select_video_btn.clicked.connect(self.select_video)
        video_layout.addWidget(select_video_btn)

        detect_video_btn = QPushButton("Dynamic Recognition")
        detect_video_btn.setObjectName("detectButton")
        detect_video_btn.clicked.connect(lambda: self.analyze_video(enable_tracking=False))
        video_layout.addWidget(detect_video_btn)
        track_video_btn = QPushButton("Path analysis")
        track_video_btn.setObjectName("analyzeButton")
        track_video_btn.clicked.connect(lambda: self.analyze_video(enable_tracking=True))
        video_layout.addWidget(track_video_btn)
        stop_analysis_btn = QPushButton("Stop analysis")
        stop_analysis_btn.clicked.connect(self.stop_video_analysis)
        video_layout.addWidget(stop_analysis_btn)
        control_layout.addWidget(video_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)

        result_group = QGroupBox("")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(0, 0, 0, 10)

        result_title = QLabel("-Test results-")
        result_title.setObjectName("groupTitle")
        result_title.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(result_title)
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(200)
        result_layout.addWidget(self.result_text)
        control_layout.addWidget(result_group)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setSpacing(15)

        original_frame = QFrame()
        original_frame.setObjectName("imageFrame")
        original_layout = QVBoxLayout(original_frame)
        original_layout.setContentsMargins(0, 0, 0, 0)
        original_label = QLabel("Original image")
        original_label.setObjectName("sectionLabel")
        original_layout.addWidget(original_label)
        self.original_view = QGraphicsView()
        self.original_view.setMinimumSize(400, 300)
        original_layout.addWidget(self.original_view)
        display_layout.addWidget(original_frame)

        processed_frame = QFrame()
        processed_frame.setObjectName("imageFrame")
        processed_layout = QVBoxLayout(processed_frame)
        processed_layout.setContentsMargins(0, 0, 0, 0)
        processed_label = QLabel("Processing results")
        processed_label.setObjectName("sectionLabel")
        processed_layout.addWidget(processed_label)
        self.processed_view = QGraphicsView()
        self.processed_view.setMinimumSize(400, 300)
        processed_layout.addWidget(self.processed_view)
        display_layout.addWidget(processed_frame)
        main_layout.addWidget(display_widget, 1)
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select drone aerial images", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.current_image = path
            self.image_label.setText(os.path.basename(path))
            self.image_label.setStyleSheet(
                "color: #2c3e50; background-color: #e0ffe0; padding: 8px; border-radius: 4px;")

            self.show_image(path, self.original_view)
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select drone aerial video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.current_video = path
            self.video_label.setText(os.path.basename(path))
            self.video_label.setStyleSheet(
                "color: #2c3e50; background-color: #e0ffe0; padding: 8px; border-radius: 4px;")
    def analyze_image(self):
        if not self.current_image:
            QMessageBox.warning(self, "提示", "Please select a picture first")
            return
        if not self.model:
            QMessageBox.warning(self, "错误", "Model not loaded correctly")
            return

        self.image_processor = ImageProcessor(self.model)

        self.image_processor.image_processed.connect(self.update_image_results)
        self.image_processor.analysis_finished.connect(self.image_analysis_finished)

        self.image_thread = QThread()
        self.image_processor.moveToThread(self.image_thread)
        self.image_thread.started.connect(lambda: self.image_processor.process_image(self.current_image))
        self.image_thread.finished.connect(self.image_thread.deleteLater)
        self.image_thread.start()
    def analyze_video(self, enable_tracking=False):
        if not self.current_video:
            QMessageBox.warning(self, "提示", "Please select a video first")
            return
        if not self.model:
            QMessageBox.warning(self, "错误", "Model not loaded correctly")
            return

        self.video_processor = VideoProcessor(self.model, self.current_video, enable_tracking)

        self.video_processor.frame_processed.connect(self.update_video_frames)
        self.video_processor.progress_updated.connect(self.update_progress)
        self.video_processor.analysis_finished.connect(self.video_analysis_finished)
        self.video_processor.stats_updated.connect(self.update_stats)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.video_thread = QThread()
        self.video_processor.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.video_processor.process_video)
        self.video_thread.finished.connect(self.video_thread.deleteLater)
        self.video_thread.start()
    def stop_video_analysis(self):
        if self.video_processor:
            self.video_processor.stop_processing()
            self.progress_bar.setVisible(False)
    def show_image(self, image_path, view):
        pixmap = QPixmap(image_path)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
    def update_image_results(self, original_image, processed_image, stats):

        self.show_frame(original_image, self.original_view)

        self.show_frame(processed_image, self.processed_view)

        self.result_text.setPlainText(json.dumps(stats, indent=2, ensure_ascii=False))
    def update_video_frames(self, original_frame, processed_frame, detections):

        self.show_frame(original_frame, self.original_view)

        self.show_frame(processed_frame, self.processed_view)
    def show_frame(self, frame, view):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap)
        view.setScene(scene)
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    def video_analysis_finished(self, message):
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "完成", message)
    def image_analysis_finished(self, message):
        QMessageBox.information(self, "完成", message)
    def update_stats(self, stats):
        self.result_text.setPlainText(json.dumps(stats, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    app = QApplication(sys.argv)

    login_dialog = LoginDialog()
    if login_dialog.exec() == QDialog.Accepted:

        window = MainWindow(login_dialog.username)
        window.show()
        sys.exit(app.exec())
    else:

        sys.exit(0)
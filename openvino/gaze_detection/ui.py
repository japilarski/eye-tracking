from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import sys
import cv2
import qimage2ndarray
import threading

from openvino.gaze_detection.gaze_estimation import Main
from openvino.gaze_detection.Calibration import CalibrationWindow
from openvino.gaze_detection.CheckerboardCalibration import CheckerboardCalibration
from openvino.gaze_detection._utils import get_screen_size
from postprocessing.pp_utils import *


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Eye Tracking")
        self.main_layout = QGridLayout()
        self.video_size = QSize(320, 240)
        self.setup_ui()

        self.checkerboard_calibration = CheckerboardCalibration()
        self.checkerboard_calibrated = False

        self.gaze_estim = Main(self.checkerboard_calibration, self.video_size.width(), self.video_size.height())
        self.is_tracking_enabled = False
        self.setup_camera_tracking()

        threading.Thread(target=self.gaze_estim.main, daemon = True).start()

        self.points_number = 0
        self.pp_points = []

    def setup_ui(self):
        self.title_label = QLabel()
        self.title_label.setText("Webcam Eye Tracking")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label, 0, 0, 1, 3)

        position_layout = QVBoxLayout()

        self.enable_postprocessing = QCheckBox("Enable postprocessing")
        position_layout.addWidget(self.enable_postprocessing)

        postprocessing_points_layout = QHBoxLayout()
        self.postprocessing_points_label = QLabel("Number of points:")
        postprocessing_points_layout.addWidget(self.postprocessing_points_label)
        self.postprocessing_points_number = QSpinBox()
        self.postprocessing_points_number.setRange(100, 3000)
        self.postprocessing_points_number.setValue(1000)
        postprocessing_points_layout.addWidget(self.postprocessing_points_number)
        position_layout.addLayout(postprocessing_points_layout)

        position_layout.addStretch(1)

        self.checkerboard_label = QLabel(
            '''<a href='https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png'>Get Checkerboard</a>''')
        self.checkerboard_label.setOpenExternalLinks(True)

        position_layout.addWidget(self.checkerboard_label)

        self.checkerboard_calibration_button = QPushButton("Start Camera Calibration")
        self.checkerboard_calibration_button.clicked.connect(self.camera_calibration_btn_handler)
        position_layout.addWidget(self.checkerboard_calibration_button)

        self.main_layout.addLayout(position_layout, 1, 0)

        buttons_layout = QHBoxLayout()

        camera_layout = QGridLayout()

        self.calibrate_button = QPushButton("Start Calibration")
        self.calibrate_button.clicked.connect(self.start_calibration)
        buttons_layout.addWidget(self.calibrate_button)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(self.video_size)
        camera_layout.addWidget(self.camera_label, 0, 0, 1, 2)
        self.main_layout.addLayout(camera_layout, 1, 1, 2, 1)

        quit_layout = QHBoxLayout()
        quit_layout.addStretch(1)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        quit_layout.addWidget(self.quit_button)

        self.track_button = QPushButton("Eye tracking")
        self.track_button.clicked.connect(self.enable_tracking)
        buttons_layout.addWidget(self.track_button)

        self.postprocess_button = QPushButton("Postprocessing")
        self.postprocess_button.clicked.connect(self.open_postprocessing)
        buttons_layout.addWidget(self.postprocess_button)
        self.main_layout.addLayout(buttons_layout, 3, 0, 1, 3)
        self.main_layout.addLayout(quit_layout, 4, 0, 1, 3)

        self.setLayout(self.main_layout)

    def setup_camera_tracking(self):
        # self.cam = cv2.VideoCapture(0)
        # self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        # self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())
        self.cam = self.gaze_estim.cam

        self.timer = QTimer()
        self.timer.timeout.connect(self.video_and_tracking)
        self.timer.start(40)

    def camera_calibration_btn_handler(self):
        if not self.checkerboard_calibration.is_calibrating:
            self.checkerboard_calibrated = True
            self.checkerboard_calibration.reset()
            self.checkerboard_calibration_button.setText("Stop Camera Calibration")

            self.calibrate_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.postprocess_button.setEnabled(False)

            self.checkerboard_calibration.start_calibration()
        else:
            self.checkerboard_calibration.stop_calibration()

            self.calibrate_button.setEnabled(True)
            self.track_button.setEnabled(True)
            self.postprocess_button.setEnabled(True)
            self.checkerboard_calibration_button.setText("Start Camera Calibration")

        return

    def start_calibration(self):
        self.calibration_window = CalibrationWindow(self.gaze_estim)
        self.calibration_window.show()
        pass

    def enable_tracking(self):
        if self.enable_postprocessing.isChecked():
            self.pp_points.clear()
            self.points_number = self.postprocessing_points_number.value()
        self.tracking_window = TrackerWindow()
        self.tracking_window.show()
        self.is_tracking_enabled = True
        # self.nn_worker.start()
        self.tracking_window.destroyed.connect(lambda: self.tracking_enabled(False))
    
    def tracking_enabled(self, state):
        self.is_tracking_enabled = state

    def collect_points(self, point):
        if len(self.pp_points) < self.points_number:
            self.pp_points.append(point)
        else:
            self.pp_points.pop(0)
            self.pp_points.append(point)

    def open_postprocessing(self):
        if len(self.pp_points) != 0:
            self.postprocessing_window = PostprocessingWindow(self.pp_points)
            self.postprocessing_window.show()
        else:
            dialog_box = QMessageBox(self)
            dialog_box.setWindowTitle("Error!")
            dialog_box.setText("No points to process! Enable postprocessing and track your eyes first.")
            dialog_box.setStandardButtons(QMessageBox.Ok)
            dialog_box.setIcon(QMessageBox.Critical)
            button = dialog_box.exec()

            if button == QMessageBox.Ok:
                self.enable_postprocessing.setChecked(True)

    def video_and_tracking(self):
        try:
            if self.is_tracking_enabled:
                self.tracking_window.moveTracker(self.gaze_estim.gaze_centre[0], self.gaze_estim.gaze_centre[1])
                if self.enable_postprocessing.isChecked():
                    self.collect_points(self.gaze_estim.gaze_centre)

            frame = self.gaze_estim.result_img

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.flip(frame, 1)
            image = qimage2ndarray.array2qimage(frame)  # SOLUTION FOR MEMORY LEAK
            self.camera_label.setPixmap(QPixmap.fromImage(image))
        except:
            #print("Waiting for image")
            pass


class TrackerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking...")
        self.h, self.w = get_screen_size()
        self.setup_ui()

    def setup_ui(self):
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowState(Qt.WindowFullScreen)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(self.w, self.h)

        self.tracker_label = QLabel(self)
        self.tracker = QPixmap('kursor.png')
        self.tracker_label.setPixmap(self.tracker)
        self.tracker_label.setFixedSize(100, 100)

        self.tracker_label.move(int(self.w / 2 - 50), int(self.h / 2 - 50))
        pass

    def moveTracker(self, x, y):
        self.tracker_label.move(x - 50, y - 50)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def wheelEvent(self, event: QWheelEvent):
        pass


class PostprocessingWindow(QWidget):
    def __init__(self, saved_points):
        super().__init__()
        self.setWindowTitle("Postprocessing")
        self.h, self.w = get_screen_size()
        self.setAttribute(Qt.WA_DeleteOnClose)
        # self.setFixedSize(self.w, self.h)
        self.points = saved_points
        self.create_scene()

    def create_scene(self):
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, self.w, self.h)
        self.view = QGraphicsView(self.scene, self)
        self.view.setGeometry(0, 0, self.w, self.h)
        self.thresh = int(self.w / 10)   # 10% szerokosci ekranu

        self.fixations, self.saccades = detect_fixations_saccades(self.points, self.thresh)

        print(f"Fixations: {(self.fixations)}")
        print(f"Saccades: {(self.saccades)}")

        for fixation in self.fixations:
            self.add_fixation(fixation[0], fixation[1])

        for i in range(len(self.saccades) - 1):
            self.add_saccade(self.saccades[i], self.saccades[i+1])

        self.view.show()

    def add_fixation(self, x, y):
        self.scene.addEllipse(x, y, 10, 10, QPen(Qt.red), QBrush(Qt.red))

    def add_saccade(self, p1, p2):
        self.scene.addLine(p1[0], p1[1], p2[0], p2[1], QPen(Qt.blue))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    pass

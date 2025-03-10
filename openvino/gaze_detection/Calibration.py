from enum import Enum

import numpy as np
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from openvino.gaze_detection._utils import get_screen_size


class CalibrationState(Enum):
    INITIALIZING = 1
    PREPARING = 2
    ACTIVE = 3
    DONE = 4


class CalibrationDot(QGraphicsItemGroup):
    def __init__(self, x, y, scale_factor):
        super().__init__()
        x = int(x / scale_factor)  # windows dpi scalling fix
        y = int(y / scale_factor)
        self.green_dot = QGraphicsEllipseItem(x-10, y-10, 40, 40)
        self.green_dot.setBrush(QBrush(Qt.green))
        self.green_dot.setVisible(False)
        self.black_dot = QGraphicsEllipseItem(x, y, 20, 20)
        self.black_dot.setBrush(QBrush(Qt.black))

        self.addToGroup(self.green_dot)
        self.addToGroup(self.black_dot)

    def show_green_dot(self):
        self.green_dot.setVisible(True)


class CalibrationWindow(QGraphicsView):
    calibration_results_signal = Signal(object)

    def __init__(self,gaze_estim):
        super().__init__()
        self.gaze_estim = gaze_estim
        self.setWindowTitle("Calibration")
        self.setWindowState(Qt.WindowFullScreen)
        self.calibration_layout = QGridLayout()
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scale_factor = self.devicePixelRatio()
        self.h, self.w = get_screen_size()

        self.dots_positions = [
            (self.w * 0.1, self.h * 0.1),
            (self.w * 0.5, self.h * 0.1),
            (self.w * 0.9, self.h * 0.1),
            (self.w * 0.1, self.h * 0.5),
            (self.w * 0.5, self.h * 0.5),
            (self.w * 0.9, self.h * 0.5),
            (self.w * 0.1, self.h * 0.9),
            (self.w * 0.5, self.h * 0.9),
            (self.w * 0.9, self.h * 0.9),
        ]
        self.dots = [CalibrationDot(x, y, self.scale_factor) for x, y in self.dots_positions]

        self.state = CalibrationState.INITIALIZING

        self.current_dot = None
        self.current_dot_idx = 0

        self.dots_predictions = []
        self.predictions = []
        self.no_take_gaze_points = 9
        self.which_gaze_point = 0

        self.setup_ui()
        QTimer.singleShot(0, self._run_calibration_step)

    def setup_ui(self):
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setSceneRect(0, 0, int(self.w), int(self.h))

    def wheelEvent(self, event: QWheelEvent):
        pass

    def add_dot(self, dot: CalibrationDot):
        self.scene.addItem(dot)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            try:
                self.nn_subscription.disconnect(self.nn_prediction_handler)
            except:
                pass
            self.close()

    def _run_calibration_step(self, new_state=None):
        if new_state is not None:
            self.state = new_state

        if self.state == CalibrationState.INITIALIZING:
            QTimer.singleShot(1_000, lambda: self._run_calibration_step(CalibrationState.PREPARING))

        elif self.state == CalibrationState.PREPARING:
            if self.current_dot is not None:
                print(str(self.predictions))
                self.dots_predictions.append(np.mean(self.predictions, axis=0))
                print("self.dots_predictions"+str(self.dots_predictions))
                self.predictions = []
                self.which_gaze_point = 0
                self.current_dot_idx += 1

            if len(self.dots) > self.current_dot_idx:
                self.scene.clear()
                self.current_dot = self.dots[self.current_dot_idx]
                self.add_dot(self.current_dot)
                QTimer.singleShot(3_000, lambda: self._run_calibration_step(CalibrationState.ACTIVE))
            else:
                self._run_calibration_step(CalibrationState.DONE)

        elif self.state == CalibrationState.ACTIVE:
            self.predictions.append(self.gaze_estim.gaze_centre)
            self.current_dot.show_green_dot()
            # 9 times getting the gaze coords, the goal is to calculate mean so it will be more accurate
            if (self.no_take_gaze_points>= self.which_gaze_point):
                self.which_gaze_point +=1
                print("OK")
                QTimer.singleShot(100, lambda: self._run_calibration_step(CalibrationState.ACTIVE))

            else:
                QTimer.singleShot(5_000, lambda: self._run_calibration_step(CalibrationState.PREPARING))

        elif self.state == CalibrationState.DONE:
            self.close()
            #Calibration parameterers calculation
            print()
            print(self.dots_predictions[0][0])
            self.deltas_calib_all = []

            for iter,item in enumerate(self.dots_positions):
                    print(item)
                    delta_x = item[0] - self.dots_predictions[iter][0]
                    print("item0"+str(item[0])+"- calib_list_all"+str(self.dots_predictions[iter][0]))
                    delta_y = item[1] - self.dots_predictions[iter][1]
                    print("iter"+str(iter)+"delta_x"+str(delta_x)+"delta_y"+str(delta_y))
                    
                    if len(self.deltas_calib_all)<9:
                        self.deltas_calib_all.append((delta_x,delta_y))
                
            delta_x_mean = 0
            delta_y_mean = 0
            for item in self.deltas_calib_all:
                delta_x_mean += item[0]
                delta_y_mean += item[1]

            delta_x_mean = delta_x_mean/len(self.deltas_calib_all)
            delta_y_mean = delta_y_mean/len(self.deltas_calib_all)
            print("delta_x_mean"+str(delta_x_mean))
            print("delta_y_mean"+str(delta_y_mean))
            return

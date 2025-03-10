import cv2
import numpy as np
import json
import os


class CheckerboardCalibration:
    def __init__(self):
        self.frames = []
        self.grid_size = (9, 6)
        self.calibrations_results_filepath = "./calibration_matrix.json"
        self.data = None
        self.is_calibrating = False

        self.load_stored_calibration_results()

    def load_stored_calibration_results(self):
        if os.path.exists(self.calibrations_results_filepath):
            with open(self.calibrations_results_filepath, 'r') as json_file:
                self.data = json.load(json_file)
                self.data['rms'] = np.array(self.data['rms'])
                self.data['camera_matrix'] = np.array(self.data['camera_matrix'])
                self.data['dist_coeff'] = np.array(self.data['dist_coeff'])

    def start_calibration(self):
        self.is_calibrating = True

    def stop_calibration(self):
        self.is_calibrating = False
        self.calibrate()

    def reset(self):
        self.is_calibrating = False
        self.frames = []

    def add_frame(self, img):
        self.frames.append(img.copy())

    def calibrate(self):
        self.is_calibrating = False
        if len(self.frames) == 0:
            return

        x, y = self.grid_size

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((y * x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        samples = 0
        for img in self.frames:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

            if ret:
                samples += 1
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                if samples >= 20:
                    break

        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        self.data = {
            'rms': np.asarray(rms).tolist(),
            'camera_matrix': np.asarray(mtx).tolist(),
            'dist_coeff': np.asarray(dist).tolist()
        }

        with open(self.calibrations_results_filepath, 'w') as json_file:
            json.dump(self.data, json_file)

        self.data['rms'] = np.array(self.data['rms'])
        self.data['camera_matrix'] = np.array(self.data['camera_matrix'])
        self.data['dist_coeff'] = np.array(self.data['dist_coeff'])

        self.reset()

    def undistort_image(self, img):
        if self.data is None:
            return img

        camera_matrix = self.data['camera_matrix']
        dist = self.data['dist_coeff']

        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist, None, new_camera_matrix)

        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst

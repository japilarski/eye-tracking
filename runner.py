import os

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from face_detection import RetinaFace
from screeninfo import get_monitors
from l2cs import L2CS
from torch import nn

from transforms import VectorTransformer


class L2CS_Runner:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_tests', 'l2cs', 'models', 'L2CSNet_gaze360.pkl')

        self._is_running = False

        self._device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._model = None
        self._detector = None
        self._softmax = None
        self._idx_tensor = None
        self._transformations = None
        self._capture = None

        self._initialized = False

    def initialize(self, capture=None):
        if not self._initialized:
            state_dict = torch.load(self.model_path, map_location=self._device)

            self._model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()

            self._softmax = nn.Softmax(dim=1)
            self._detector = RetinaFace(gpu_id=0 if torch.cuda.is_available() else -1)

            self._idx_tensor = torch.FloatTensor(list(range(90))).to(self._device)
            self._transformations = transforms.Compose([
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            self._initialized = True
            print('Model Initialized')

            self._capture = capture

    def run(self, emit_prediction=None):
        self._ensure_initialized()

        if self._is_running:
            return

        if self._capture is None:
            self._capture = cv2.VideoCapture(0)
            if not self._capture.isOpened():
                raise IOError("Cannot open webcam")

        main_monitor = next((monitor for monitor in get_monitors() if monitor.is_primary), None)
        if not main_monitor:
            raise Exception("Somehow could not fetch monitor specs")

        screen_resolution = np.array([main_monitor.height, main_monitor.width])
        screen_width = main_monitor.width_mm / 1000

        # this stuff is hardcoded and won't work with different setup
        eye_position = np.array([-0.02, 0.0, 0.55])
        camera_offset = np.array([0.0, screen_width / 2, 0.0]) + np.array([-0.025, 0.0, 0.025])

        gaze_tracker = VectorTransformer(eye_position, camera_offset, screen_width, screen_resolution)

        self._is_running = True

        with torch.no_grad():
            while self._is_running:
                success, frame = self._capture.read()
                face = self._detect_face(frame)

                if face is not None:
                    pitch, yaw = self._predict_pitch_yaw(face)
                    gaze_vect = self._gaze_to_cartesian(pitch, yaw)  # pitch, yaw -> x y z
                    gaze_vect_camera = self._transform_gaze_coord_to_camera_coord(gaze_vect)  # head coordinate system -> camera coordinate system
                    #print(f"Pitch: {pitch}, Yaw: {yaw}, XYZ: {gaze_vect_camera}")
                    # gaze_tracker.visualize(gaze_vect_camera)
                    xy_preds_screen = gaze_tracker.compute_pixel_coordinates(gaze_vect_camera)
                    #print('screen coords', xy_preds_screen)

                    data = {
                        'screen_coords': xy_preds_screen,
                        'gaze_angles': [pitch, yaw]
                    }

                    if emit_prediction is not None:
                        emit_prediction(data)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    def stop(self):
        self._is_running = False

    def _detect_face(self, frame):
        faces = self._detector(frame)
        if faces is None or len(faces) == 0:
            return None

        faces.sort(key=lambda face: face[2], reverse=True)
        box, landmarks, score = faces[0]

        if score < .95:
            return None
        x_min = int(box[0])
        if x_min < 0:
            x_min = 0
        y_min = int(box[1])
        if y_min < 0:
            y_min = 0
        x_max = int(box[2])
        y_max = int(box[3])

        img = frame[y_min:y_max, x_min:x_max]
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        img = self._transformations(im_pil)
        img = Variable(img).to(self._device)
        img = img.unsqueeze(0)

        return img

    def _predict_pitch_yaw(self, img):
        gaze_pitch, gaze_yaw = self._model(img)

        pitch_predicted = self._softmax(gaze_pitch)
        yaw_predicted = self._softmax(gaze_yaw)

        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self._idx_tensor) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self._idx_tensor) * 4 - 180

        pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

        return pitch_predicted, yaw_predicted

    def _ensure_initialized(self):
        if not self._initialized:
            raise Exception("Model is not initialized")

    def _gaze_to_cartesian(self, pitch, yaw):
        gaze_xyz = np.array([
            -np.cos(yaw) * np.sin(pitch),
            -np.sin(yaw),
            -np.cos(yaw) * np.cos(pitch)
        ])
        return gaze_xyz

    def _transform_gaze_coord_to_camera_coord(self, gaze_vect_l2cs):
        return np.array([
            gaze_vect_l2cs[1],
            -gaze_vect_l2cs[0],
            gaze_vect_l2cs[2],
        ])


if __name__ == "__main__":
    runner = L2CS_Runner()
    runner.initialize()
    runner.run()
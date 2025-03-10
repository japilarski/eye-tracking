import numpy as np
import matplotlib.pyplot as plt


class VectorTransformer:
    """
        A class to transform gaze vectors into pixel coordinates on a screen.

        Attributes
        ----------
        eye_position : numpy.ndarray
            The 3D position of the eye.
        camera_position : numpy.ndarray
            The 3D position of the camera.
        screen_resolution : numpy.ndarray
            The resolution of the screen in pixels.
        aspect_ratio : float
            The aspect ratio of the screen.
        screen_height : float
            The height of the screen in meters.
        camera_offset : numpy.ndarray
            The 3D offset of the camera from the up left corner of the screen
        """
    def __init__(self, eye_position, camera_offset, screen_width, screen_resolution, camera_position=np.array([0.0, 0.0, 0.0])):
        """
                Initializes the VectorTransformer with the given parameters.

                Parameters
                ----------
                eye_position : list or tuple
                    The 3D position of the eye.
                camera_position : list or tuple
                    The 3D position of the camera.
                screen_width : float
                    The width of the screen in meters.
                screen_resolution : list or tuple
                    The resolution of the screen in pixels.
                """
        self.eye_position = np.array(eye_position)
        self.camera_position = np.array(camera_position)
        self.screen_resolution = np.array(screen_resolution)
        self.aspect_ratio = self.screen_resolution[1] / self.screen_resolution[0]
        self.screen_width = screen_width
        self.screen_height = self.screen_width / self.aspect_ratio
        self.camera_offset = camera_offset

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.scatter(*self.eye_position, color='blue', s=100, label='Eye')
        self.ax.scatter(*self.camera_position, color='red', s=100, label='Camera')

        self.screen_corners = [
            self.camera_position - self.camera_offset,
            self.camera_position - self.camera_offset + [self.screen_height, 0, 0],
            self.camera_position - self.camera_offset + [self.screen_height, self.screen_width, 0],
            self.camera_position - self.camera_offset + [0, self.screen_width, 0],
        ]
        self.screen_corners.append(self.screen_corners[0])
        self.screen_corners = np.array(self.screen_corners)
        self.ax.plot(self.screen_corners[:, 0], self.screen_corners[:, 1],self. screen_corners[:, 2], color='black')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()

    def set_eye_position_from_vector(self, eye_to_camera_vector):
        """
        Sets the eye position based on a vector from the eye to the camera.

        Parameters
        ----------
        eye_to_camera_vector : list or tuple
            The 3D vector from the eye to the camera.
        """
        self.eye_position = self.camera_position + np.array(eye_to_camera_vector)

    def _compute_intersection(self, eye_position, gaze_vector, screen_corners):
        """
                Computes the intersection of the gaze vector with the screen plane.

                Parameters
                ----------
                eye_position : numpy.ndarray
                    The 3D position of the eye.
                gaze_vector : numpy.ndarray
                    The 3D gaze vector.
                screen_corners : list of numpy.ndarray
                    The 3D positions of the corners of the screen.

                Returns
                -------
                numpy.ndarray
                    The 3D position of the intersection point.
                """
        n = np.array([0, 0, 1])
        d = -np.dot(n, screen_corners[0])
        t = -(np.dot(n, eye_position) + d) / np.dot(n, gaze_vector)
        intersection = eye_position + t * gaze_vector
        return intersection

    def _point_to_pixel(self, intersection, screen_corners):
        """
                Converts a 3D intersection point to 2D pixel coordinates on the screen.

                Parameters
                ----------
                intersection : numpy.ndarray
                    The 3D position of the intersection point.
                screen_corners : list of numpy.ndarray
                    The 3D positions of the corners of the screen.

                Returns
                -------
                tuple of int
                    The x and y pixel coordinates on the screen.
                """
        height_vector = screen_corners[1] - screen_corners[0]
        width_vector = screen_corners[3] - screen_corners[0]
        u = np.dot(intersection - screen_corners[0], width_vector) / np.linalg.norm(width_vector)**2
        v = np.dot(intersection - screen_corners[0], height_vector) / np.linalg.norm(height_vector)**2
        pixel_x = u * self.screen_resolution[1]
        pixel_y = v * self.screen_resolution[0]

        pixel_x = np.clip(pixel_x, 0, self.screen_resolution[1])
        pixel_y = np.clip(pixel_y, 0, self.screen_resolution[0])
        # print(f'Pixel coordinates on the screen: {int(pixel_x), int(pixel_y)}')
        return int(pixel_x), int(pixel_y)

    def compute_pixel_coordinates(self, gaze_vector):
        """
                Computes the pixel coordinates on the screen based on the gaze vector.

                Parameters
                ----------
                gaze_vector : list or tuple
                    The 3D gaze vector.

                Returns
                -------
                tuple of int
                    The x and y pixel coordinates on the screen.
                """
        screen_corners = [
            self.camera_position - self.camera_offset,
            self.camera_position - self.camera_offset + [self.screen_height, 0, 0],
            self.camera_position - self.camera_offset + [self.screen_height, self.screen_width, 0],
            self.camera_position - self.camera_offset + [0, self.screen_width, 0],
        ]
        intersection = self._compute_intersection(self.eye_position, gaze_vector, screen_corners)
        return self._point_to_pixel(intersection, screen_corners)

    def visualize(self, gaze_vector,calibration):
        """
                Visualizes the eye, camera, gaze vector, and screen plane.

                Parameters
                ----------
                gaze_vector : list or tuple
                    The 3D gaze vector.
                """
        self.ax.clear()
        self.ax.scatter(*self.eye_position, color='blue', s=100, label='Eye')
        self.ax.scatter(*self.camera_position, color='red', s=100, label='Camera')
        self.ax.quiver(*self.eye_position, *gaze_vector, length=0.65, color='green', label='Gaze Vector')

        self.ax.plot(self.screen_corners[:, 0], self.screen_corners[:, 1], self.screen_corners[:, 2], color='black')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()

        if calibration==True:
            plt.close()
        else:
            plt.draw()
        plt.pause(0.001)
        #plt.show()

if __name__ == "__main__":
    # example values
    #calibration=True
    #eye_position = np.array([-0.10, 0.0, 0.3])
    eye_position = np.array([0.30, 0.0, 0.3])
    gaze_vector = np.array([0.4, -0.3, -1.0])
    #gaze_vector = np.array([0.4, -0.0, -1.0])
    screen_width = 0.6  # in meters
    camera_offset = np.array([0.0, screen_width / 2, 0.0])
    screen_resolution = np.array([1080, 1920])

    gaze_tracker = VectorTransformer(eye_position, camera_offset, screen_width, screen_resolution)
    pixel_coordinates = gaze_tracker.compute_pixel_coordinates(gaze_vector)

    gaze_tracker.visualize(gaze_vector)

    eye_to_camera_vector = [0.1, 0, 0.4]
    gaze_tracker.set_eye_position_from_vector(eye_to_camera_vector)
    gaze_vector = np.array([0.5, 0.6, -1.0])
    pixel_coordinates = gaze_tracker.compute_pixel_coordinates(gaze_vector)
    gaze_tracker.visualize(gaze_vector)
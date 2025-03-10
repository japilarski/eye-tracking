# EYE - Eye Tracking

## Project assumptions

- manual calibration - relative camera position will be input by the user in the GUI before the program starts
- a quick calibration test will be performed to determine the postion and size of the screen
- a neural network will be used to estimate the gaze vector and head position
- using the gaze vector to estimate the point on the screen that the user is looking at 
- the application should be able to work in a well-lit room and no hard shadows covering the face
- the user should be positioned 40-80 centimeters from the camera
- the application works for a single user in camera's view with the user's eyes and face visible and not covered,
e. g. by a mask or glasses.
- minimum camera resolution: 720p
- recording the series of estimated points at which the user was looking (fixations and saccades) and calculation user
statistics, showing gaze route etc.


## Expected final results

- Mean Angular Error: 12 degrees


## Tech stack

- **language:** Python 3.10
- **libraries:**
    - PyTorch - neural network building
    - OpenCV - image processing
    - NumPy, Jax - calculations
    - PySide6 - GUI

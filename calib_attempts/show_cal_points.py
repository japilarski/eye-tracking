import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.linear_model import LinearRegression


def read_calibration_data(file_path):
    gaze_vectors = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Gaze Vector (Camera Coordinates):" in line:
                values = re.findall(r'[-.\d]+', line)
                gaze_vector = np.array([float(value) for value in values[3:]])
                gaze_vectors.append(gaze_vector)

    print("gaze vectors",gaze_vectors)
    return gaze_vectors

def plot_calibration_points(gaze_vectors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, gaze_vector in enumerate(gaze_vectors):
        ax.scatter(gaze_vector[0], gaze_vector[1], gaze_vector[2], label=f'Point {i+1}')

    ax.set_xlabel('Gaze Point (X)')
    ax.set_ylabel('Gaze Point  (Y)')
    ax.set_zlabel('Gaze Point  (Z)')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_path = 'calibration_data.txt'
    calib_gaze_points = read_calibration_data(file_path)
    plot_calibration_points(calib_gaze_points)
    # Measured data
    measured_points = np.array([
        [-0.04566591, -0.21599953, -0.97435544],#1
        [-0.14830042, 0.05933918, -0.98658548],# 2
        [-0.02549459, 0.34207303, -0.9352874],# 3
        [0.13523987, -0.23766492, -0.96154355],#4
        [0.13152595, 0.10517462, -0.98517344],#5
        [0.09897412, 0.46148928, -0.8805818],#6
        [0.33202847, -0.19381679, -0.92285697],#7
        [0.39418734, 0.15473161, -0.90486869],#8
        [0.32964552, 0.55787953, -0.75986678],# 9
        [-0.04689521, -0.21893683, -0.97405712],#1
        [-0.17815018, 0.14925963, -0.97164546],#2
        [0.0408714, 0.47367209, -0.87827864],# 3
        [0.13539745, -0.15088463, -0.97908739],#4
        [0.02683853, 0.08476352, -0.99597694],#5
        [0.16845721, 0.54099426, -0.82276438],#6
        [0.28019314, -0.06521538, -0.95619437],#7
        [0.39957185, 0.27501263, -0.87407119],#8
        [0.2986399, 0.55653088, -0.77490781]#9
    ])
    measured_points2 = np.array([
        [-0.04689521, -0.21893683, -0.97405712],#1
        [-0.17815018,  0.14925963, -0.97164546],#2
        [ 0.0408714 ,  0.47367209 ,-0.87827864],#3
        [ 0.13539745, -0.15088463 ,-0.97908739],#4
        [ 0.02683853,  0.08476352 ,-0.99597694],#5
        [ 0.16845721,  0.54099426 ,-0.82276438],#6
        [ 0.28019314, -0.06521538 ,-0.95619437],#7
        [ 0.39957185,  0.27501263 ,-0.87407119],#8
        [ 0.2986399,   0.55653088, -0.77490781]#9
    ])
    # measured_points = np.array([
    #     [-0.10055834, -0.29521374, -0.94887294],
    #     [-0.11698357, 0.02395701, - 0.99250682],
    #     [-0.13675226, 0.52203051, -0.84111489],
    #     [0.05355347, -0.2065733, -0.97676852],
    #     [0.04483705, 0.07855979, -0.99572371],
    #     [0.09576919, 0.56641903, - 0.81670626],
    #     [0.31778225, -0.08169486, -0.94376062],
    #     [0.38824537, 0.22499052, -0.89224327],
    #     [0.31620834, 0.49086587, -0.81104148]
    # ])
    # Expected data
    expected_points = np.array([
        [-0.025, -0.3, -0.99],#1
        [-0.025, 0.0, -0.994],#2
        [-0.025, 0.3, -0.99],#3
        [0.1825, -0.3, -0.994],#4
        [0.1825, 0.0, -0.998],#5
        [0.1825, 0.3, -0.994],#6
        [0.34, -0.3, -0.99],#7
        [0.34, 0.0, -0.994],#8
        [0.34, 0.3, -0.99]#9
    ])
    # Linear regression model
    regression_model = LinearRegression()

    regression_model.fit(calib_gaze_points, expected_points)
    #regression_model.fit(measured_points, expected_points)

    # Constrains
    x_min, x_max = -0.025, 0.34
    y_min, y_max = -0.3, 0.3
    z_min, z_max = -1, -0.92
    # x_min, x_max = -0.1, 0.6
    # y_min, y_max = -0.485, 0.6
    # z_min, z_max = -1, -0.92

    calibrated_gaze_vectors=[]
    for i, gaze_vector in enumerate(calib_gaze_points):
        calibrated_gaze_vect = regression_model.predict(gaze_vector.reshape(1, -1))
        calibrated_gaze_vect = calibrated_gaze_vect.reshape(-1)
        # Applying constrains and checking how they affect correction
        # calibrated_gaze_vect[0] = max(x_min, min(x_max, calibrated_gaze_vect[0]))
        # calibrated_gaze_vect[1] = max(y_min, min(y_max, calibrated_gaze_vect[1]))
        # calibrated_gaze_vect[2] = max(z_min, min(z_max, calibrated_gaze_vect[2]))

        calibrated_gaze_vectors.append(calibrated_gaze_vect)
    plot_calibration_points(calibrated_gaze_vectors)

    # Showing predicted values
    print("Przewidziane wartości:")
    print(calibrated_gaze_vectors)

    # Exptected values
    print("\nWartości oczekiwane:")
    print(expected_points)

    # Regression coefficients
    print("\nWspółczynniki regresji:")
    print("Współczynniki nachylenia:", regression_model.coef_)
    print("Intercept:", regression_model.intercept_)

    punkty_testowe = [-0.04566591, -0.21599953, -0.97435544],
    print("test")
    predicted_points = regression_model.predict(punkty_testowe)
    print(predicted_points)

import cv2
import numpy as np
import pickle

cap = cv2.VideoCapture(0)  # Otwórz dostęp do kamery (numer 0 to zazwyczaj kamera domyślna)

# Odczytaj rozmiar ramek obrazów
frame_width = int(cap.get(3))  # Szerokość ramki
frame_height = int(cap.get(4))  # Wysokość ramki

print(f"Rozmiar ramek obrazów: {frame_width}x{frame_height}")

cap.release()  # Zwolnij dostęp do kamery

# Wczytaj dane kalibracyjne kamery z plików .pkl
with open('cameraMatrix.pkl', 'rb') as file:
    cameraMatrix = pickle.load(file)

with open('dist.pkl', 'rb') as file:
    dist = pickle.load(file)

# Rozmiar obrazu z kamery (ustawiony zgodnie z wymaganiami kamery)
frameSize = (frame_width, frame_height)

# Oblicz nową macierz kamery i mapy remappingu
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, frameSize, 1, frameSize)
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, frameSize, cv2.CV_32FC1)

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)  # 0 oznacza użycie domyślnej kamery (możesz zmienić na inny indeks, jeśli masz więcej kamer)

while True:
    ret, frame = cap.read()  # Odczytaj obraz z kamery

    if not ret:
        break

    # Wykonaj remapping obrazu
    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    # Przycinanie do obszaru ROI
    # x, y, w, h = roi
    # undistorted_frame = undistorted_frame[y:y + h, x:x + w]

    # Wyświetl obraz na żywo
    cv2.imshow('Undistorted Camera Feed', undistorted_frame)

    # Przerwanie pętli po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby i zamknij okna
cap.release()
cv2.destroyAllWindows()

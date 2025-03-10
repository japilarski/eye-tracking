import cv2
import numpy as np
from openvino.runtime import Core
from model_api.performance_metrics import PerformanceMetrics
from time import perf_counter

import openvino as ov

# TO DO JUTRO:
# dodac gaze detection model
# i inpucik z kamery

# initialize OpenVINO API 
core = ov.Core()
detection_model_xml = "C:\\Users\\julka\\Studia\\ACiRsem7\\ZSD\\OpenVINO\\openvino_detector_2022_3\\openvino_detector_2022_3\\model_2022_3\\face-detection-retail-0005.xml"
detection_model = core.read_model(model = detection_model_xml)
device = "CPU" # choose processor as device

# compile model for specific device
compiled_model = core.compile_model(model = detection_model, device_name = device)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


cap = cv2.VideoCapture(0)
N, C, H, W = input_layer.shape
print("H"+str(H))
print("W"+str(W))
metrics = PerformanceMetrics()

while True:
    start_time = perf_counter()
    ret, frame = cap.read()  # read a frame
    #resize frame so it can fit into model
    print("H"+str(H))
    print("W"+str(W))
    resized_image = cv2.resize(src = frame, dsize = (W, H))
    
    input_data = np.expand_dims(np.transpose(resized_image, (2,0,1)), 0).astype(np.float32)
    request = compiled_model.create_infer_request()
    request.infer(inputs = {input_layer.any_name:input_data})
    result = request.get_output_tensor(output_layer.index).data

    bboxes = []
    frame_height, frame_width = frame.shape[:2]
    for detection in result[0][0]:
        lebel = int(detection[1])
        conf = float(detection[2])       # confidence score
        if conf >0.76:
            xmin = int(detection[3]*frame_width)
            ymin = int(detection[4]*frame_height)
            xmax = int(detection[5]*frame_width)
            ymax = int(detection[6]*frame_height)
            bboxes.append((xmin,ymin,xmax, ymax))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 255, 255), 3)
            cv2.putText(frame, "face", (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)
    
    metrics.update(start_time,frame)
    cv2.imshow("Person detection", frame)
    key = cv2.waitKey(1)

    if key in {ord('q'), ord('Q'), 27}:
        cap.release()
        cv2.destroyAllWindows()
        break



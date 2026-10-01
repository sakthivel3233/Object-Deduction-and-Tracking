import os
from tkinter import filedialog

from imageai.Detection import ObjectDetection
obj_detect = ObjectDetection()
obj_detect.setModelTypeAsYOLOv3()
obj_detect.setModelPath("./models/yolo-tiny.h5")
obj_detect.loadModel()
import cv2
file_path = filedialog.askopenfilename()
file_name = os.path.basename(file_path)
print("FilePath=" + file_name)
cam_feed = cv2.VideoCapture(file_name)
cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)
while True:
    ret, img = cam_feed.read()
    annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=img,
                                                               input_type="array",
                                                               output_type="array",
                                                               display_percentage_probability=False,
                                                               display_object_name=True)

    cv2.imshow("", annotated_image)

    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
        break

cam_feed.release()
cv2.destroyAllWindows()
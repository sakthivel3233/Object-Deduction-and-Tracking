from tkinter import filedialog

import urllib3
#pip install tensorflow==2.4.0
#pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0
#pip install imageai --upgrade
urllib3.disable_warnings()
import os
from imageai.Detection import ObjectDetection


execution_path = os.getcwd()
file_path = filedialog.askopenfilename()
file_name = os.path.basename(file_path)
print("FilePath=" + file_name)
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "yolo5.h5"))
detector.loadModel()
#detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image1.jpeg"), output_image_path=os.path.join(execution_path , "output_3.jpg"))
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , file_path), output_image_path=os.path.join(execution_path , "output/"+file_name))
count = 0
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


print(detections)
import cv2
import matplotlib.pyplot as plt
img_bgr = cv2.imread(os.path.join(execution_path , "output/"+file_name), 1)
plt.imshow(img_bgr)
plt.title("Output Image")
plt.show()
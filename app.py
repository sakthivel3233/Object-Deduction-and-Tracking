from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, Response
app = Flask(__name__)
from tkinter import filedialog
import urllib3
import cv2
import matplotlib.pyplot as plt
#pip install tensorflow==2.4.0
#pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0
#pip install imageai --upgrade
urllib3.disable_warnings()
import os
from imageai.Detection import ObjectDetection
path_python ='C:\ProgramData\Microsoft\Windows\Start Menu\Programs\JetBrains'

MY_FOLDER = os.path.join('static','images')


@app.route('/')
def home():
    msg = ''

    return render_template('home.html', msg=msg)

@app.route('/images',methods=['GET','POST'])
def images():
    msg = 'green.jpg'
    filename=''
    file_name=''
    file_name1 = ''
    if request.method == 'POST':
        f = request.files['file']
        msg=f.filename
        filename=f.filename

        f.save(secure_filename(f.filename))

        img = cv2.imread('static/images/' + filename)
        execution_path = os.getcwd()
        file_name = 'static/images/' + filename
        file_name1 = 'static/images/out_' + filename
        print("FilePath=" + file_name)
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(execution_path, "yolo5.h5"))
        detector.loadModel()
        # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image1.jpeg"), output_image_path=os.path.join(execution_path , "output_3.jpg"))
        detections = detector.detectObjectsFromImage(input_image=file_name,output_image_path=file_name1)
        count = 0
        for eachObject in detections:
            print(eachObject["name"], " : ", eachObject["percentage_probability"])

        print(detections)
        print(file_name)
        print(file_name1)
    return render_template('image.html', msg=msg, output=file_name1)


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == "POST":
        if 'videoInput' not in request.files:
            return "No file part", 400
        file = request.files['videoInput']
        if file.filename == '':
            return "No selected file", 400
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            obj_detect = ObjectDetection()
            obj_detect.setModelTypeAsYOLOv3()
            obj_detect.setModelPath("./models/yolo.h5")
            obj_detect.loadModel()

            def generate_frames():
                cam_feed = cv2.VideoCapture(filename)
                if not cam_feed.isOpened():
                    yield f"Error: Unable to open video file {filename}".encode()
                    return

                while True:
                    ret, img = cam_feed.read()
                    if not ret:
                        break
                    annotated_image, preds = obj_detect.detectObjectsFromImage(
                        input_image=img,
                        input_type="array",
                        output_type="array",
                        display_percentage_probability=False,
                        display_object_name=True
                    )
                    ret, buffer = cv2.imencode('.jpg', annotated_image)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                cam_feed.release()

            return Response(generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('video.html')

@app.route('/webcam',methods=['GET','POST'])
def webcam():
    msg = ''
    filename= ''
    import os
    os.system(
        path_python + 'python object_web.py')
    if request.method=="POST":
        from imageai.Detection import ObjectDetection
        obj_detect = ObjectDetection()
        obj_detect.setModelTypeAsYOLOv3()
        obj_detect.setModelPath("./models/yolo.h5")
        obj_detect.loadModel()
        import cv2

        cam_feed = cv2.VideoCapture(0)
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

    return render_template('start.html', msg=msg)


if __name__ == '__main__':
    app.run(port=5000,debug=True)
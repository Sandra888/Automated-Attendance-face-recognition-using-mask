from sklearn.metrics       import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm           import SVC
from imutils.face_utils    import FaceAligner
from imutils.video         import VideoStream
from imutils               import paths
from keras_facenet         import FaceNet
from flask                 import render_template
from flask                 import Response
from flask                 import jsonify
from flask                 import request
from flask                 import Flask
from os                    import path as csv_file_finder
from os                    import listdir

import cv2 as opencv
import pandas as pd
import numpy as np
import threading
import datetime
import imutils
import pickle
import time
import dlib
import csv
import os

"""Variables"""
video_frame            = None
student_name           = None
student_rollno         = None
student_wear_mask      = None
current_module         = None
status_message_main    = None
recognized_face_table  = None
attendance_table       = None
sub_attendance_table   = "<table class='table'><thead><tr><th scope='col'><div class='float-left d-inline-block'><a href='#' id='attendance_backbtn' onclick='ViewAttendance()'><i class='material-icons' style='vertical-align:middle;font-size:19px;color: white;'>keyboard_backspace</i></a></div>View Attendance</th></tr></thead><tbody><tr><td><div style='height:38vh; padding-top: 8px;'><table class='table-dark table-bordered compressed'><thead><tr><th style='width: 4%;'>S.no</th><th style='width: 40%;'>Name</th><th style='width: 20%;'>Rollno</th><th>Date</th><th>Time</th></tr></thead><tbody></tbody></table></div></td></tr> </tbody></table>" 
temp_predicted_names   = []
label_path             = "C:/Users/MicroMedia/Desktop/Project_A_v2/model/datasets/label.csv"
testing_path           = "C:/Users/MicroMedia/Desktop/Project_A_v2/model/datasets/test/"
training_path          = "C:/Users/MicroMedia/Desktop/Project_A_v2/model/datasets/train/"
label_encoder_path     = "C:/Users/MicroMedia/Desktop/Project_A_v2/model/output/label_encoder.pickle"
recognizer_model_path  = "C:/Users/MicroMedia/Desktop/Project_A_v2/model/output/recognizer_model.pickle"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
resnet_ssd_model_path  = "C:/Users/MicroMedia/Desktop/Project_A_v2/model/premade/face_detection_model/deploy.prototxt"
resnet_ssd_weight_path = "C:/Users/MicroMedia/Desktop/Project_A_v2/model/premade/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"


"""Objects"""
app                       = Flask(__name__)
facenet                   = FaceNet()
resnet_ssd                = opencv.dnn.readNetFromCaffe(resnet_ssd_model_path, resnet_ssd_weight_path)
dlib_front_face_detector  = dlib.get_frontal_face_detector()
dlib_front_face_landmarks = dlib.shape_predictor("C:/Users/MicroMedia/Desktop/Project_A_v2/model/premade/dlib_model/shape_predictor_68_face_landmarks.dat")
dlib_front_face_alignment = FaceAligner(dlib_front_face_landmarks, desiredFaceWidth=256)


"""Routes"""
@app.route("/")
def frontend():
    return render_template("index.html")

@app.route("/get_video_frame")
def get_video_frame():
    return Response(generate_video_frame(),mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/capture_module/",methods = ['POST'])
def capture_module():
    global student_name, student_rollno, student_wear_mask, current_module	
    student_name                = request.form["student_name"]
    student_rollno              = request.form["student_rollno"]
    student_wear_mask           = request.form["student_wear_mask"]
    current_module              = "capture_module"
    create_status_message("Starting Capture and Train Module",0)
    create_recognized_face_table([])
    return "success"

@app.route("/attendance_module/",methods = ['POST'])
def attendance_module():
    global current_module, session
    create_status_message("Starting Session Module",0)
    current_module              = "attendance_module"
    session                     = bool(int(request.form["data"]))
    if session :
        create_recognized_face_table([])
    return "success"

@app.route('/get_status_message/',methods=['POST'])
def get_status_message():
    global status_message_main
    return jsonify({'data': status_message_main})

@app.route('/get_recognized_face_table/',methods=['POST'])
def get_recognized_face_table():
    global recognized_face_table
    return jsonify({'data': recognized_face_table})

@app.route('/retrive_sub_attendance_module/',methods=['POST'])
def retrive_sub_attendance_module():
    global sub_attendance_table
    sub_attendance_table ="<table class='table'><thead><tr><th scope='col'><div class='float-left d-inline-block'><a href='#' id='attendance_backbtn' onclick='ViewAttendance()'><i class='material-icons' style='vertical-align:middle;font-size:19px;color: white;'>keyboard_backspace</i></a></div>View Attendance</th></tr></thead><tbody><tr><td><div style='height:38vh; padding-top: 8px;'><table class='table-dark table-bordered compressed'><thead><tr><th style='width: 4%;'>S.no</th><th style='width: 40%;'>Name</th><th style='width: 20%;'>Rollno</th><th>Date</th><th>Time</th></tr></thead><tbody></tbody></table></div></td></tr> </tbody></table>" 
    sub_create_attendance_table(request.form["csv_id"])
    return "success"

@app.route('/get_attendance_module/',methods=['POST'])
def get_attendance_module():
    global attendance_table
    return jsonify({'data': attendance_table})

@app.route('/get_sub_attendance_module/',methods=['POST'])
def get_sub_attendance_module():
    global sub_attendance_table
    return jsonify({'data': sub_attendance_table})

"""Functions"""
def generate_video_frame():
    global video_frame
    while True:
        if video_frame is None:
            continue
        (flag, encodedImage) = opencv.imencode(".jpg", video_frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

def backend():
    global video_frame, student_name, student_rollno, student_wear_mask, current_module, training_path, testing_path, recognizer_model_path, label_encoder_path, session
    create_status_message("None",0)
    create_recognized_face_table([])
    create_attendance_table()
    while True:
        """Capturing Module"""
        if current_module                == "capture_module":
            gif_camera_turning_on = opencv.VideoCapture('C:/Users/MicroMedia/Desktop/Project_A_v2/static/image/camera_turning_on.gif')
            while True:
                okay, gif_frame = gif_camera_turning_on.read()
                if not okay :
                    break
                video_frame     = gif_frame.copy()
                time.sleep(0.1)
            video_stream                  = VideoStream(src=0).start()
            time.sleep(2.0)
            student_to_be_inserted        = [student_rollno , student_name]
            student_availability_in_lable = 1
            with open(label_path, mode = 'r') as lable_file_r:
                lable_file_reader         = csv.DictReader(lable_file_r)
                for row in lable_file_reader:
                    if row["Name"] == student_name :
                        student_availability_in_lable = 0
                lable_file_r.close()
            if(bool(student_availability_in_lable)):
                with open(label_path,'a+') as lable_file_a:
                    lable_file_writer = csv.writer(lable_file_a)
                    lable_file_writer.writerow(student_to_be_inserted)
                    lable_file_a.close()
            temp_testing_path   = testing_path  + student_name
            temp_training_path  = training_path + student_name
            if not os.path.exists(temp_training_path):
                os.mkdir(temp_testing_path)
                os.mkdir(temp_training_path) 
            temp_testing_path   = temp_testing_path +"/"
            temp_training_path  = temp_training_path+"/"
            current_file_no_captured = 1
            while current_file_no_captured <= 100:
                frame                      = video_stream.read()
                temp_frame                 = frame.copy()
                temp_frame                 = imutils.resize(temp_frame.copy(), width=600)
                (image_height,image_width) = temp_frame.shape[:2]
                face_detections            = resnet_ssd_face_detection(temp_frame)
                if len(face_detections) > 0:
                    max_confidence_index   = np.argmax(face_detections[0, 0, :, 2])
                    max_confidence         = face_detections[0, 0, max_confidence_index, 2]
                    if max_confidence > 0.5:
                        bounding_box = face_detections[0, 0, max_confidence_index, 3:7] * np.array([image_width, image_height, image_width, image_height])
                        (bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height) = bounding_box.astype("int")
                        cropped_face = temp_frame[bounding_box_y:bounding_box_height, bounding_box_x:bounding_box_width]
                        aligned_face = dlib_face_alignment(cropped_face)
                        if aligned_face is None:
                            aligned_face = temp_frame
                        if(current_file_no_captured <= 80):
                            opencv.imwrite(temp_training_path+str(student_wear_mask)+str(current_file_no_captured)+".png",aligned_face)
                        else:
                            opencv.imwrite(temp_testing_path +str(student_wear_mask)+str(current_file_no_captured)+".png",aligned_face)
                        opencv.rectangle(temp_frame, (bounding_box_x, bounding_box_y), (bounding_box_width, bounding_box_height),(200, 92, 0), 2)
                        video_frame  = temp_frame.copy()
                        create_status_message("Capturing Faces "+str(current_file_no_captured)+"% [1/4]",int(current_file_no_captured/4))
                        current_file_no_captured = current_file_no_captured + 1
                    else:
                        video_frame  = frame.copy()
                else:   
                    video_frame      = frame.copy() 
            video_stream.stop()
            gif_camera_turning_off = opencv.VideoCapture('C:/Users/MicroMedia/Desktop/Project_A_v2/static/image/camera_turning_off.gif')
            while True:
                okay, gif_frame = gif_camera_turning_off.read()
                if not okay :
                    break
                video_frame     = gif_frame.copy()
                time.sleep(0.1)
            frame = opencv.imread("C:/Users/MicroMedia/Desktop/Project_A_v2/static/image/no_camera.png")
            video_frame = frame.copy()
            """Extracting Module"""

            create_status_message("Extracting Embeddings 0% [2/4]",25)
            training_paths  = list(paths.list_images(training_path))
            embeddings = []
            names      = []
            images     = []
            for path in training_paths: 
                name  = path.split(os.path.sep)[-2]
                image = opencv.imread(path)
                names.append(name)
                images.append(image)
            bulk_embeddings = facenet_embeddings(images)
            for index,embedding in enumerate(bulk_embeddings):
                embeddings.append(embedding.flatten())
                create_status_message("Extracting Embeddings "+str(int(((index+1)/len(bulk_embeddings))*100))+"% [2/4]",(25+int(((index/len(bulk_embeddings))*100)/4)))
            time.sleep(1.0)

            """Training Module"""
            create_status_message("Training model 0% [3/4]",50)
            training_data = {"embeddings": embeddings, "names": names}
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(training_data["names"])
            recognizer_model = SVC(C=1.0, kernel="linear", probability=True)
            recognizer_model.fit(training_data["embeddings"], labels)
            create_status_message("Training model 100% [3/4]",75)
            f = open(recognizer_model_path, "wb")
            f.write(pickle.dumps(recognizer_model))
            f.close()
            f = open(label_encoder_path, "wb")
            f.write(pickle.dumps(label_encoder))
            f.close()
            time.sleep(1.0)

            """Testing Module"""
            create_status_message("Testing model 0% [4/4]",75)
            testing_paths  = list(paths.list_images(testing_path))
            names      = []
            images     = []
            correctly_predicted_probabilities = 0
            for path in testing_paths: 
                name  = path.split(os.path.sep)[-2]
                image = opencv.imread(path)
                names.append(name)
                images.append(image)
            bulk_embeddings = facenet_embeddings(images)
            
            for index,embedding in enumerate(bulk_embeddings):
                recognizer_model_input=[]
                recognizer_model_input.append(embedding)
                predicted_probabilities = recognizer_model.predict_proba(recognizer_model_input)[0]
                highest_predicted_probability_index = np.argmax(predicted_probabilities)
                predicted_class = label_encoder.classes_[highest_predicted_probability_index]
                if str(names[index]) == str(predicted_class) :
                    correctly_predicted_probabilities = correctly_predicted_probabilities+1
                create_status_message("Testing model "+str(int(((index+1)/len(bulk_embeddings))*100))+"% [4/4]",75+int(((index/len(bulk_embeddings))*100)/4))
            time.sleep(1.0)
            create_status_message("Model successfully trained and tested with an accuracy of "+str(int((correctly_predicted_probabilities/len(bulk_embeddings))*100))+"%","success")
            current_module = None
        elif current_module == "attendance_module":

            """attendance_module"""
            gif_camera_turning_on = opencv.VideoCapture('C:/Users/MicroMedia/Desktop/Project_A_v2/static/image/camera_turning_on.gif')
            while True:
                okay, gif_frame = gif_camera_turning_on.read()
                if not okay :
                    break
                video_frame     = gif_frame.copy()
                time.sleep(0.1)
            video_stream    = VideoStream(src=0).start()
            time.sleep(2.0)
            recognizer_model   = pickle.loads(open(recognizer_model_path, "rb").read())
            label_encoder      = pickle.loads(open(label_encoder_path, "rb").read())
            database_names     = pd.read_csv(label_path)
            database_col_names = ['Id','Name','Date','Time']
            attendance_sheet   = pd.DataFrame(columns = database_col_names) 
           
            create_status_message("Session Running...","running")
            while session :
                frame                      = video_stream.read()
                temp_frame                 = frame.copy()
                temp_frame                 = imutils.resize(temp_frame.copy(), width=600)
                (image_height,image_width) = temp_frame.shape[:2]
                face_detections            = resnet_ssd_face_detection(temp_frame)
                predicted_classes          = []
                recognized_face_table_row  = []
                at_least_one_detection = 0
                detection_index = []
                for confidence_index in  range(0, face_detections.shape[2]):
                    confidence         = face_detections[0, 0, confidence_index, 2]
                    if confidence > 0.5:
                        at_least_one_detection = 1
                        detection_index.append(confidence_index)
                        bounding_box = face_detections[0, 0, confidence_index, 3:7] * np.array([image_width, image_height, image_width, image_height])
                        (bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height) = bounding_box.astype("int")
                        cropped_face = temp_frame[bounding_box_y:bounding_box_height, bounding_box_x:bounding_box_width]
                        aligned_face = dlib_face_alignment(cropped_face)
                        if aligned_face is None:
                            aligned_face = temp_frame
                        facenet_embeddings_input = []
                        facenet_embeddings_input.append(aligned_face)

                        embeddings = facenet_embeddings(facenet_embeddings_input)
                        predicted_probabilities = recognizer_model.predict_proba(embeddings)[0]
                        highest_predicted_probability_index = np.argmax(predicted_probabilities)
                        actual_value_y=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                        print("Confusion Matrix")
                        print(confusion_matrix(actual_value_y, predicted_probabilities.ravel()))
                        predicted_class = label_encoder.classes_[highest_predicted_probability_index]    
                        prediction_probability = predicted_probabilities[highest_predicted_probability_index]
                        if int(prediction_probability*100)<10:
                            predicted_class = "Unknown"
                        predicted_classes.append(predicted_class)
                if bool(at_least_one_detection):
                    optimize_predicted_class("insert",predicted_classes)
                    optimized_names = optimize_predicted_class("get")  
                    for current_detection_index in range(len(detection_index)):
                        bounding_box = face_detections[0, 0, detection_index[current_detection_index], 3:7] * np.array([image_width, image_height, image_width, image_height])
                        (bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height) = bounding_box.astype("int")
                        corrected_bounding_box_y = bounding_box_y - 10 if bounding_box_y - 10 > 10 else bounding_box_y + 10
                        if(optimized_names[current_detection_index]=="Recognizing..."):
                            opencv.rectangle(temp_frame, (bounding_box_x, bounding_box_y), (bounding_box_width, bounding_box_height),(200, 92, 0), 2)
                            opencv.putText(temp_frame, str(optimized_names[current_detection_index]), (bounding_box_x, corrected_bounding_box_y),opencv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 92, 0), 2)
                        elif(optimized_names[current_detection_index]=="Unknown"):
                            unknown_image_paths = list(paths.list_images("unknown/"))
                            current_no_unknown = len(unknown_image_paths)+1
                            cv2.imwrite("unknown/"+str(current_no_unknown)+".png",frame)
                        else:
                            current_time = time.time()      
                            current_date = datetime.datetime.fromtimestamp(current_time).strftime('%d-%m-%Y')
                            current_time_stamp = datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')
                            predicted_rollno = database_names.loc[database_names['Name'] == optimized_names[current_detection_index]]['Id'].values
                            attendance_sheet.loc[len(attendance_sheet)] = [predicted_rollno,optimized_names[current_detection_index],current_date,current_time_stamp]
                            recognized_face_row =[optimized_names[current_detection_index],predicted_rollno]
                            recognized_face_table_row.append(recognized_face_row)
                            opencv.rectangle(temp_frame, (bounding_box_x, bounding_box_y), (bounding_box_width, bounding_box_height),(0, 200, 0), 2)
                            opencv.putText(temp_frame, str(optimized_names[current_detection_index]), (bounding_box_x, corrected_bounding_box_y),opencv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                video_frame = temp_frame.copy()
                create_recognized_face_table(recognized_face_table_row)
            video_stream.stop()
            create_status_message("Session Running...","running")
            gif_camera_turning_off = opencv.VideoCapture('C:/Users/MicroMedia/Desktop/Project_A_v2/static/image/camera_turning_off.gif')
            while True:
                okay, gif_frame = gif_camera_turning_off.read()
                if not okay :
                    break
                video_frame     = gif_frame.copy()
                time.sleep(0.1)
            frame = opencv.imread("C:/Users/MicroMedia/Desktop/Project_A_v2/static/image/no_camera.png")
            video_frame = frame.copy()
            create_recognized_face_table([])
            attendance_sheet = attendance_sheet[~attendance_sheet["Name"].apply(tuple).duplicated()]
            current_time = time.time()      
            current_date = datetime.datetime.fromtimestamp(current_time).strftime('%d-%m-%Y')
            if (csv_file_finder.exists("C:/Users/MicroMedia/Desktop/Project_A_v2/attendance/"+current_date+".csv")):
                with open("C:/Users/MicroMedia/Desktop/Project_A_v2/attendance/"+current_date+".csv", mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    line_count = 0
                    for row in csv_reader:
                        attendance_sheet.loc[len(attendance_sheet)]=[row["Id"],row["Name"],row["Date"],row["Time"]]
                csv_file.close()
                attendance_sheet = attendance_sheet[~attendance_sheet["Name"].apply(tuple).duplicated()]
            csv_file_name = "C:/Users/MicroMedia/Desktop/Project_A_v2/attendance/"+current_date+".csv"
            attendance_sheet.to_csv(csv_file_name, index=False)
            create_status_message("Session Saved to csv.","success")
            create_attendance_table()
            current_module = None
        else:        
            frame = opencv.imread("C:/Users/MicroMedia/Desktop/Project_A_v2/static/image/no_camera.png")
            video_frame = frame.copy()

def optimize_predicted_class(module,predicted_name=""):
    global temp_predicted_names
    if module == "insert":
        if len(temp_predicted_names) == 0:
            temp_predicted_names.append(predicted_name)
        else:
            if temp_predicted_names[-1]==predicted_name:
                if len(temp_predicted_names) >=5:
                    temp_predicted_names.pop()
                    temp_predicted_names.append(predicted_name)
                else:
                    temp_predicted_names.append(predicted_name)
            else :
                temp_predicted_names=[]
                temp_predicted_names.append(predicted_name)
    else :
        if len(temp_predicted_names)<5:
            temp_recognition = []
            for no_of_predections in range(len(temp_predicted_names[-1])):
                temp_recognition.append("Recognizing...")
            return temp_recognition
        else :                
            return temp_predicted_names[-1]
        
def resnet_ssd_face_detection(temp_frame):
    global resnet_ssd
    imageBlob = opencv.dnn.blobFromImage(opencv.resize(temp_frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
    resnet_ssd.setInput(imageBlob)
    temp_face_detections = resnet_ssd.forward()
    return temp_face_detections

def dlib_face_alignment(temp_cropped_face):
    global dlib_front_face_detector, dlib_front_face_landmarks, dlib_front_face_alignment
    try:
        gray_temp_cropped_face    = opencv.cvtColor(temp_cropped_face, opencv.COLOR_BGR2GRAY)
    except:
        temp_cropped_face = None
        return temp_cropped_face 
    else:
        dlib_bounding_box         = dlib_front_face_detector(gray_temp_cropped_face, 2)
        if len(dlib_bounding_box)!= 0:
            temp_aligned_face      = dlib_front_face_alignment.align(temp_cropped_face,gray_temp_cropped_face,dlib_bounding_box[0])  
            return temp_aligned_face
        return temp_cropped_face
    

def facenet_embeddings(temp_images):
    return facenet.embeddings(temp_images)

def create_status_message(message,percentage_completed):
    global status_message_main
    status_message_main=""
    if percentage_completed =="success":
        status_message_main =   "<div style='padding-bottom:10px;'> \
                                    Status: "+message+"\
                                </div>\
                                <div class='progress mx-auto' style='height: 1px; width: 40%;'>\
                                    <div class='progress-bar progress-bar-striped bg-success progress-bar-animated' role='progressbar' style='width: 100%' aria-valuenow='100' aria-valuemin='0' aria-valuemax='100'>\
                                    </div>\
                                </div>"  
    elif percentage_completed =="running":
        status_message_main =   "<div style='padding-bottom:10px;'> \
                                    Status: "+message+"\
                                </div>\
                                <div class='progress mx-auto' style='height: 1px; width: 40%;'>\
                                    <div class='progress-bar progress-bar-striped bg-danger progress-bar-animated' role='progressbar' style='width: 100%' aria-valuenow='100' aria-valuemin='0' aria-valuemax='100'>\
                                    </div>\
                                </div>"                
    else :
        status_message_main =   "<div style='padding-bottom:10px;'> \
                                    Status: "+message+"\
                                </div>\
                                <div class='progress mx-auto' style='height: 1px; width: 40%;'>\
                                    <div class='progress-bar progress-bar-striped progress-bar-animated' role='progressbar' style='width: "+str(percentage_completed)+"%' aria-valuenow='100' aria-valuemin='0' aria-valuemax='100'>\
                                    </div>\
                                </div>"

def create_recognized_face_table(table_rows):
    global recognized_face_table
    if len(table_rows) == 0:
        recognized_face_table = " "
    else:
        recognized_face_table = " "
        current_row = 1
        recognized_face_table = "<table class='table-dark table-bordered xcompresse'><tbody>"
        for rows in table_rows :
            recognized_face_table_row = "<tr> <td style='width: 2%;'>"+str(current_row)+"</td> <td style='width: 40%;'>"+str(rows[0])+"</td> <td style='width: 30%;'>"+str(rows[1])+"</td>  </tr>"  
            recognized_face_table = recognized_face_table + recognized_face_table_row
            current_row = current_row+1
        recognized_face_table = recognized_face_table +"</tbody></table>"

def create_attendance_table():
    global attendance_table
    attendance_table=""
    cvs_files = []
    file_names = listdir("C:/Users/MicroMedia/Desktop/Project_A_v2/attendance/")
    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_name = file_name.split(".")
            cvs_file = str(file_name[0])
            cvs_files.append(cvs_file)
    attendance_table="<table class='table-dark table-bordered compressed'><thead><tr><th style='width: 3%;'>S.no</th><th style='width: 50%;'>Date (dd-mm-yyyy)</th><th style='width: 20%;'>View</th></tr></thead><tbody>"
    for csv_file_index in range(len(cvs_files)):
        attendance_table_row="<tr><td>"+str(int(csv_file_index)+1)+"</td><td>"+cvs_files[csv_file_index]+"</td><td><button class='btn btn-secondary view_sub_attendance_module' value='"+cvs_files[csv_file_index]+"' type='submit' style='padding:1px; background-color: rgb(33, 37, 41);border: none;'><i class='material-icons ' style='vertical-align:middle;font-size:19px;color: white; padding:1px;'>assignment</i></button></td></tr>"
        attendance_table=attendance_table+attendance_table_row
    attendance_table = attendance_table +"</tbody></table>"   

def sub_create_attendance_table(csv_id):
    global sub_attendance_table
    sub_attendance_table = ""
    read_sub_csv = pd.read_csv("C:/Users/MicroMedia/Desktop/Project_A_v2/attendance/"+csv_id+'.csv')
    sub_attendance_table="<table class='table'><thead><tr><th scope='col'><div class='float-left d-inline-block'><a href='#' id='attendance_backbtn' onclick='ViewAttendance()'><i class='material-icons' style='vertical-align:middle;font-size:19px;color: white;'>keyboard_backspace</i></a></div>"+str(csv_id)+"</th></tr></thead><tbody><tr><td><div style='height:38vh; padding-top: 8px;'><table class='table-dark table-bordered compressed'><thead><tr><th style='width: 4%;'>S.no</th><th style='width: 40%;'>Name</th><th style='width: 20%;'>Rollno</th><th>Date</th><th>Time</th></tr></thead><tbody>"
    for read_index in range(len(read_sub_csv)):
        read_sub_row = "<tr><td>"+str(int(read_index)+1)+"</td><td>"+str(read_sub_csv['Name'][read_index])+"</td><td>"+str(read_sub_csv['Id'][read_index])+"</td><td>"+str(read_sub_csv['Date'][read_index])+"</td><td>"+str(read_sub_csv['Time'][read_index])+"</td></tr>"
        sub_attendance_table = sub_attendance_table+read_sub_row
    sub_attendance_table = sub_attendance_table +"</tbody></table></div></td></tr> </tbody></table>"   
      

"""Program starts here"""
if __name__ == '__main__' :
    thread = threading.Thread(target=backend)
    thread.daemon = True
    thread.start()
    app.run(host="localhost", port="8888", debug=True,threaded=True, use_reloader=False)

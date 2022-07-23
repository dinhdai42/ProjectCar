import cv2 as cv
import numpy as np
import imutils
from imutils.video import FPS
from playsound import playsound
import multiprocessing as mp
from pygame import mixer
mixer.init()


# Distance constants
KNOWN_DISTANCE = 3  # meters
NGUOIDIBO_WIDTH = 0.4  # INCHES
XEMAYNGANG_WIDTH = 1.9  # meters
XEMAYDOC_WIDTH = 0.60 #meters

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("yolo.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setting up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny-custom_final_v2.weights', 'yolov4-tiny-custom_v2.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(412, 412), scale=1 / 255, swapRB=True)


# object detector funciton /method

def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 1)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 1)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # person class id
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        elif classid == 1:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        elif classid == 2:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance




# reading the reference image from dir
ref_person = cv.imread('q.jpg')
ref_person = cv.resize (ref_person,(640,480))
#ref_mobile = cv.imread('ReferenceImages/image5.png')
#ref_motorbike = cv.imread('Wave_rear_Calib.png')

#mobile_data = object_detector(ref_mobile)
#mobile_width_in_rf = mobile_data[0][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

#motorbike_data = object_detector(ref_motorbike)
#motorbike_width_in_rf = motorbike_data[0][1]
xemaydoc_width_in_rf = 90
xemayngang_width_in_rf=300
nguoidibo_width_in_rf=100
print(f" motorbike width in pixel: {person_width_in_rf}  ")

# finding focal length
focal_nguoidibo = focal_length_finder(KNOWN_DISTANCE, NGUOIDIBO_WIDTH, nguoidibo_width_in_rf)

focal_xemaydoc = focal_length_finder(KNOWN_DISTANCE, XEMAYDOC_WIDTH, xemaydoc_width_in_rf)

focal_xemayngang = focal_length_finder(KNOWN_DISTANCE, XEMAYNGANG_WIDTH, xemayngang_width_in_rf)
cap1 = cv.imread ('q.jpg')
cap1 = cv.resize(cap1,(640,480))
#cap1 = imutils.rotate(cap1,180)
data = object_detector(cap1)
#a,b,d = cap1.shape()
for z in data:
 #   print (a,b)
    if z[0] == 'nguoi di bo':
        distance = distance_finder(focal_nguoidibo, NGUOIDIBO_WIDTH, z[1])
        x, y = z[2]
    elif z[0] == 'xe may doc':
        distance = distance_finder(focal_xemaydoc, XEMAYDOC_WIDTH, z[1])
        x, y = z[2]
    elif z[0] == 'xe may ngang':
        distance = distance_finder(focal_xemayngang, XEMAYNGANG_WIDTH, z[1])
        x, y = z[2]
    cv.rectangle(cap1, (x, y - 3), (x + 150, y + 23), BLACK, -1)
    cv.putText(cap1, f'K/c: {round(25.63, 2)} m', (x + 5, y + 13), FONTS, 0.48, GREEN, 1)
    print(distance)
cv.imshow("anh", cap1)
cap = cv.VideoCapture("Dz.mp4")
zfps = FPS().start()
seconds = 5
fps = cap.get(cv.CAP_PROP_FPS)  # Gets the frames per second

multiplier = fps * seconds

while True:
    ret, frame = cap.read()
    #frame = imutils.rotate(frame,180)
    #print(cap.read())
    #frame = cv.resize (frame, (640,480))
    h,w,c = frame.shape
    #print (h,w,c)





    #f = open("a.txt", "r")
    #cameraMatrix = f.read()
    #print (cameraMatrix)
    #dist = [[-0.20727448, -0.12243194,  0.00286758,  0.00913922,  1.97528123]]bbbbbca
    #img = frame
    #h, w = img.shape[:2]
    #newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

    # Undistort
    #dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y + h, x:x + w]

    tempe = 1
    roi = frame[0:400,0:w]
    #new_img = apply_roi(frame, roi)
    frame1 = roi
    data = object_detector(frame1)
    distance = 100
    #print(data)
    for d in data:
        if d[0] == 'nguoi di bo':
            distance = distance_finder(focal_nguoidibo, NGUOIDIBO_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'xe may doc':
            distance = distance_finder(focal_xemaydoc, XEMAYDOC_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'xe may ngang':
            distance = distance_finder(focal_xemayngang, XEMAYNGANG_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv.putText(frame, f'K/c: {round(distance, 2)} m', (x + 5, y + 13), FONTS, 0.48, GREEN, 1)
        print(distance)
        #print(frame1)
       # for z in zip(distance):
            #print (z)








    #print (focal_motorbike)
    pts = np.array([[0, h], [24 * w / 51, 99 * h / 200],
                    [27 * w / 51, 99 * h / 200], [w, h],
                    ],
                   np.int32)
    pts1 = np.array([[0, h], [369 * w / 816, 103 * h / 200],
                    [447 * w / 816, 103 * h / 200], [w, h],
                     ],
                    np.int32)
    #pts = np.array([[0, h], [25 * w / 51, 20 * h / 200],
                    #[26 * w / 51, 20 * h / 200], [w, h],
                    #],
                    #np.int32)
    #pts1 = np.array([[0, h], [380 * w / 816, 30 * h / 200],
                     #[436 * w / 816, 30 * h / 200], [w, h],
                     #],
                     #np.int32)
    pts = pts.reshape((-1, 1, 2))
    pts1 = pts1.reshape((-1, 1, 2))

    isClosed = True

        # Blue color in BGR
    color = (0, 255, 255)
    color1 = (0, 0, 255)

        # Line thickness of 2 px
    thickness = 1

        # Using cv2.polylines() method
        # Draw a Blue polygon with
        # thickness of 1 px
    cv.polylines(frame, [pts], isClosed, color, thickness)
    cv.polylines(frame, [pts1], isClosed, color1, thickness)
    cv.imshow('frame', frame)
    #cv.imshow('ROI',roi)
    #print(multiplier)
    #print(fps)
    frameId = int(round(cap.get(1)))
    #print (frameId)
    #print (frameId % multiplier)
    if frameId % 40 == 0:
        if distance < 20:
            mixer.music.load("Warning.wav")
            mixer.music.play()

    #mixer.music.load("chuong.wav")
    #mixer.music.play()
    #for distance in frame
    #if (distance < 10): # in frame1[[0]]:
        #playsound("chuong.wav")
        #tempe = 0
        #cv.putText(frame, 'Nguy hiem', (100, 100), FONTS, 5, GREEN, 1)
        #mixer.music.load("chuong.wav")
        #mixer.music.play()
        #if tempe == 10:
            #tempe == -50

           # mixer.music.load("chuong.wav")
           # mixer.music.play()

    key = cv.waitKey(1)
    if key == ord('q'):
        break
    zfps.update()
zfps.stop()
print("Elasped time: {:.2f}".format(zfps.elapsed()))
print("FPS: {:.2f}".format(zfps.fps()))
cv.destroyAllWindows()
cap.release()

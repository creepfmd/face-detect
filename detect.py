# -*- coding: cp1252 -*-
import numpy as np
import cv2
import time
import math
import argparse
import uuid
from time import sleep
import os, sys
from threading import Thread
from collections import deque

""" Simulate the searching process into the photo database """

#get direction of face moving
def get_direction(pointA, pointB):
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def simulate(X):
    while 1:
        for zz in range(1, len(X) / 2):
            image = cv2.resize(X[zz * 2], (150, 150))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(image, 'searching...', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0))
            cv2.imshow("DB", image)
            cv2.waitKey(70)


def read_images(path, image_size=(100, 100)):
    """ Reads the images in a given folder

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        size: A tuple with the size 

    Returns:
        A list [X, y, folder_names]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
            folder_names: The names of the folder, so you can display it in a prediction.

    Database structure example:

    database
    |
    |----> andrea
    |        |---> 1.jpg
    |        |---> 2.jpg
    |        |---> 3.jpg
    |
    |----> chiara
    |        |
    |        |---> 1.jpg
    |        |---> 2.jpg
    |---->
    |
    """
    c = 0
    X = []  # person pic
    y = []  # label
    folder_names = []  # person name
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            name = subdirname
            folder_names.append(name)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    im = cv2.equalizeHist(im)
                    if (image_size is not None):
                        im = cv2.resize(im, image_size)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    # None
                    print "[-] Error opening " + filename

    return [X, y, folder_names]

#read single folder with faces
def read_image(path, index, image_size=(100, 100)):
    X = []
    y = []
    name = []
    subdirname = os.path.split(path)
    name.append(subdirname[1])

    for filename in os.listdir(path):
        try:
            im = cv2.imread(os.path.join(path, filename))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.equalizeHist(im)
            if (image_size is not None):
                im = cv2.resize(im, image_size)
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(index)
        except IOError, (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
        except:
            # None
            print "[-] Error opening " + filename

    return [X, y, name]


""" Print all faces found in a new window """


def showManyImages(title, faccia):
    pics = len(faccia)  # number of found faces
    # r - Maximum number of images in a row
    # c - Maximum number of images in a column
    '''if pics < 1 or pics > 12:
        return
    elif pics == 1:
        r = 1
        c = 1
        size = (250, 250)
    elif pics == 2:
        r = 2
        c = 1
        size = (150, 150)
    elif pics == 3 or pics == 4:
        r = 2
        c = 2
        size = (100, 100)
    elif pics == 5 or pics == 6:
        r = 3
        c = 2
        size = (100, 100)
    elif pics == 7 or pics == 8:
        r = 4
        c = 2
        size = (75, 75)
    else:
        r = 4
        c = 3
        size = (60, 60)'''

    ##################
    r = 4
    c = 2
    size = (100, 100)
    ##################

    width = size[0]
    height = size[1]
    # print "Creating %s-%s matrix"%(height*r, width*c)
    image = np.zeros((height * r, width * c, 3), np.uint8)
    black = (0, 0, 0)  # background color
    rgb_color = black
    color = tuple(reversed(rgb_color))
    image[:] = color

    for i in range(0, len(faccia)):
        faccia[i] = cv2.resize(faccia[i], size)
        cv2.rectangle(faccia[i], (0, 0), size, (0, 0, 0), 4)  # face edge

    k = 0
    for i in range(0, r):
        for j in range(0, c):
            # print "\tWriting into "+str(i*size[0])+" "+str(i*size[0]+size[0])+" - "+str(j*size[1])+" "+str(j*size[1]+size[1])
            try:
                image[i * size[0]:i * size[0] + size[0], j * size[0]:j * size[0] + size[0]] = faccia[k]
            except:
                None
            k = k + 1

    cv2.imshow(title, image)


####################################################################
#                        MAIN
####################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the video file")
ap.add_argument("-d", "--directory", type=str, default='E:\\tmp\\000',
	help="path to the faces library")
ap.add_argument("-b", "--buffer", type=int, default=15,
	help="max buffer size")
ap.add_argument("-i", "--image_count", type=int, default=45,
	help="count of images for each face")
ap.add_argument("-c", "--confidence", type=int, default=140,
	help="confidence level for face recognition")
args = vars(ap.parse_args())


buffer = args["buffer"]
image_count = args["image_count"]
confidence_level = args["confidence"]
root_dir = args["directory"]


# read images in the folder
[X, y, folder_names] = read_images(root_dir)

y = np.asarray(y, dtype=np.int32)

# Create the model
# model = cv2.createEigenFaceRecognizer()
model = cv2.createLBPHFaceRecognizer()
# model = cv2.createFisherFaceRecognizer()

# Learn the model
print "[*] Training..."
try:
    model.train(np.asarray(X), np.asarray(y))
    print "[+] Done"
    print "[*] Starting detection..."
except:
    print "[-] Fail"
    sys.exit()

face_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye.xml')
# eye_cascade_glasses = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('C:\opencv\data\haarcascades\haarcascade_mcs_nose.xml')


if not args.get("video", False):
    webcam = cv2.VideoCapture(0)
else:
    webcam = cv2.VideoCapture(args["video"])

# webcam.set(3,1280)
# webcam.set(4,720)
size = 2

# generate the thread "simulate" as deamon to close it when the main die
# thread = Thread(target=simulate, args=[np.asarray(X)])
# thread.daemon = True
# thread.start()

start = time.time()
count = 0

old_x = []
old_y = []
need_update = []

#fill empty values
for i in range(len(folder_names)):
    old_x.append(deque([0] * buffer, buffer))
    old_y.append(deque([0] * buffer, buffer))
    need_update.append(len(os.walk(root_dir+ "\\" +folder_names[i]).next()[2]) < image_count)

while True:
    (rval, frame) = webcam.read()

    if args.get("video") and not frame.any():
        break

    height, width, depth = frame.shape
    # if the width is too high, resize it for faster processing
    if width > 640:
        frame = cv2.resize(frame, (1024, 600))
        size = 3

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    mini = cv2.resize(gray, (gray.shape[1] / size, gray.shape[0] / size))

    # faces = face_cascade.detectMultiScale(mini, scaleFactor=1.2, minNeighbors=4, minSize=(10, 10), maxSize=(600, 600))
    # faces = face_cascade.detectMultiScale(mini,minNeighbors=4,minSize=(10, 10))
    faces = face_cascade.detectMultiScale(mini)
    faccia = []  # used to collect found faces

    #detect all faces on picture
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]

        width, height = face.shape

        # calcolo distanza
        # d = width, Z = real distance, D = real width
        # f = d*Z/D
        #
        # Z' = distanza da misurare, d' = width
        # Z' = D*f/d'
        #
        f = 202 * 52 / 18
        distance = 20 * f / width

        face_resize = cv2.resize(face, (100, 100))
        face_color = frame[y:y + h, x:x + w]
        face_color2 = face_color.copy()

        #find eyes on face
        roi_eye_gray = face[height / 5:height * 6 / 11, width * 1 / 10:width * 9 / 10]
        roi_eye_color = face_color[height / 5:height * 6 / 11, width * 1 / 10:width * 9 / 10]

        # find nose on face
        roi_nose_gray = face[height * 2 / 5:height * 8 / 11, width * 2 / 7:width * 5 / 7]
        roi_nose_color = face_color[height * 2 / 5:height * 8 / 11, width * 2 / 7:width * 5 / 7]

        # find mouth on face
        roi_mouth_gray = face[height * 7 / 11:height, width * 1 / 6:width * 5 / 6]
        roi_mouth_color = face_color[height * 7 / 11:height, width * 1 / 6:width * 5 / 6]

        # cv2.imshow('roi1',roi_eye_gray)
        # cv2.imshow('roi2',roi_nose_gray)
        # cv2.imshow('roi3',roi_mouth_gray)

        eyes = []
        eyes = eye_cascade.detectMultiScale(roi_eye_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_eye_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            # eyes_g = []
            # eyes_g = eye_cascade_glasses.detectMultiScale(roi_eye_gray)
            # for (ex,ey,ew,eh) in eyes_g:
            # cv2.rectangle(roi_eye_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
        mouth = []
        mouth = mouth_cascade.detectMultiScale(roi_mouth_gray)
        for (ex, ey, ew, eh) in mouth:
            cv2.rectangle(roi_mouth_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
        nose = []
        nose = nose_cascade.detectMultiScale(roi_nose_gray)
        for (ex, ey, ew, eh) in nose:
            cv2.rectangle(roi_nose_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

        #if found either eye, nose or mouth // or width < 65
        if len(eyes) != 0 or len(mouth) != 0 or len(nose) != 0:
            faccia.append(face_color2)
            #find face in database
            prediction = model.predict(face_resize)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if prediction[1] < confidence_level:
                cv2.putText(frame, '%s - %.0f' % (folder_names[prediction[0]], prediction[1]), (x, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                cv2.putText(frame, 'ID: %d' % (prediction[0]), (x, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                cv2.putText(frame, 'Distance: %d cm' % (distance), (x, y + h + 45), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0))
                cv2.putText(frame, 'x: %d' % x, (x, y + h + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                cv2.putText(frame, 'y: %d' % y, (x, y + h + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                old_x[prediction[0]].append(x)
                old_y[prediction[0]].append(y)

                #determine face moving direction
                direction = get_direction((old_x[prediction[0]][len(old_x[prediction[0]])-1], old_y[prediction[0]][len(old_y[prediction[0]])-1]),
                        (old_x[prediction[0]][0], old_y[prediction[0]][0]))
                cv2.putText(frame, 'Direction: %d' % direction, (x, y + h + 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                #if we need more pictures for this face then save one
                if need_update[prediction[0]]:
                    if (cv2.imwrite(root_dir + "\\" + folder_names[prediction[0]] + "\\face-" + uuid.uuid1().hex +".jpg", face_color2)):
                        cv2.imshow("Saving...", face_color2)
                        print "Face %s updated" % folder_names[prediction[0]]
                        need_update[prediction[0]] = len(os.walk(root_dir + "\\" + folder_names[prediction[0]]).next()[2]) < image_count

                        # update model
                        [new_X, new_y, new_folder_name] = read_image(root_dir + "\\" + folder_names[prediction[0]], prediction[0])

                        new_y = np.asarray(new_y, dtype=np.int32)
                        print "[*] Updating..."
                        try:
                            model.update(np.asarray(new_X), np.asarray(new_y))
                            print "[+] Done"
                            print "[*] Resuming detection after 1 second..."
                            sleep(1)
                        except:
                            print "[-] Fail"
                            sys.exit()
            else:
                #saving new face
                new_uid = uuid.uuid1()
                os.mkdir(root_dir + "\\" + new_uid.hex)
                if (cv2.imwrite(root_dir + "\\" + new_uid.hex + "\\face-first.jpg", face_color2)):
                    cv2.imshow("Saving...", face_color2)
                    print "Face %s saved" % new_uid.hex
                cv2.putText(frame, 'New face', (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                cv2.putText(frame, 'Distance: %d cm' % (distance), (x, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0))
                cv2.putText(frame, 'x: %d' % x, (x, y + h + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                cv2.putText(frame, 'y: %d' % y, (x, y + h + 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                #update model
                [new_X, new_y, new_folder_name] = read_image(root_dir + "\\" + new_uid.hex, len(folder_names))
                old_x.append(deque([0] * buffer, buffer))
                old_y.append(deque([0] * buffer, buffer))
                need_update.append(True)
                folder_names.append(new_folder_name[0])
                new_y = np.asarray(new_y, dtype=np.int32)
                print "[*] Updating..."
                try:
                    model.update(np.asarray(new_X), np.asarray(new_y))
                    print "[+] Done"
                    print "[*] Resuming detection after 1 second..."
                    sleep(1)
                except:
                    print "[-] Fail"
                    sys.exit()

            #drawing trace for found face
            for i in xrange(1, len(old_x[prediction[0]])):
                if old_x[prediction[0]][i - 1] is None or old_x[prediction[0]][i] is None:
                    continue
                thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
                cv2.line(frame, (old_x[prediction[0]][i - 1], old_y[prediction[0]][i - 1]), (old_x[prediction[0]][i], old_y[prediction[0]][i]), (0, 0, 255), thickness)


    count = count + 1
    current = time.time()
    cv2.putText(frame, " fps=%.1f" % (count / (current - start)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0))  # frame processed per second

    cv2.imshow('Video Stream', frame)
    showManyImages("Found", faccia)
    key = cv2.waitKey(1)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()

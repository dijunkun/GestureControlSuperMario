import cv2
from numpy import *
import numpy as np
import math
import operator
import keyboard
import time
quality = 16


def convert_pix(image):
    img_pix = image.copy()
    if len(img_pix) == 0 or len(img_pix[0]) == 0:
        pl = 0
        pw = 0
        ln = 1
        wn = 1
    else:
        pl = len(img_pix)
        pw = len(img_pix[0])
        ln = pl / quality + 1
        wn = pw / quality + 1

    pix = arange(quality*quality).reshape(quality, quality).reshape(quality, quality, 1)
    for px in range(0, quality):
        for py in range(0, quality):
            color = 0
            for pj in range(ln * py, ln * (py + 1)):
                for pk in range(wn * px, wn * (px + 1)):
                    if pj < pl and pk < pw:
                        color += img_pix[pj, pk]
                    else:
                        color += 0
            pix[py, px] = color / (ln * wn)
            if pix[py, px] > 127:
                pix[py, px] = 1
            else:
                pix[py, px] = 0
    return pix


def classify(cap_array, data_array, label, k):
    difference = tile(cap_array, (600, 1, 1, 1)) - data_array
    # square
    sq_diff = difference ** 2
    # get the sum
    sq_distance = sq_diff.sum(axis=1)
    sq_distance = sq_distance.sum(axis=1)
    # square root
    distance = sq_distance ** 0.5
    dis = arange(600).reshape(600)
    for ii in range(600):
        dis[ii] = distance[ii]
    # sort from smallest to biggest and return the subscript
    sort_dist = dis.argsort()
    # store the vote of labels
    class_count = {}
    # print sort_dist
    for m in range(k):
        vote_label = label[sort_dist[m]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # sort the labels
    sort_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sort_count[0][0]


face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

gesture = ['clockwise', 'anticlockwise', 'frontpalm', 'backpalm', 'holdon', 'lighta', 'lightb', 'takeoff']

key_direction = ['w', 's', 'a', 'd']
key_function = ['j', 'k']

# read sample data
np_array = arange(quality*quality*600).reshape(quality*600, quality).reshape(600, quality, quality).reshape(600, quality, quality, 1)

for index_label in range(8):
    if index_label < 4:
        for index in range(50):
            np_array[index + index_label * 50] = np.load('data/' + gesture[index_label] + str(index + 1) + '.npy')
    else:
        for index in range(100):
            np_array[index + (index_label - 4) * 100 + 200] = np.load('data/' + gesture[index_label] + str(index + 1) + '.npy')

labels = [0 for k in range(600)]
for index_label in range(8):
    if index_label < 4:
        for index in range(50):
            labels[index + index_label*50] = gesture[index_label]
    else:
        for index in range(100):
            labels[index + (index_label - 4) * 100 + 200] = gesture[index_label]

cap = cv2.VideoCapture(0)
last_result = 0
last_index_key_direction = 0
last_x = 0
last_y = 0
ref_position_x = 0
ref_position_y = 0
count = 0
while cap.isOpened():
    # read image
    ret, img = cap.read()
    imgcy = img.copy()
    # remove face
    grey1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        grey1,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(150, 150),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(imgcy, (fx, fy), (fx + fw, fy + fh), (255, 255, 255), -1)
        face_region = imgcy[fy:fy + fh, fx:fx + fw]

    MIN = np.array([0, 48, 80], np.uint8)
    MAX = np.array([20, 255, 255], np.uint8)  # HSV: V-79%
    HSVImg = cv2.cvtColor(imgcy, cv2.COLOR_BGR2HSV)
    #cv2.imshow('HSV', HSVImg)
    filterImg = cv2.inRange(HSVImg, MIN, MAX)  # filtering by skin color
    #cv2.imshow('HSVfilter', filterImg)
    filterImg = cv2.erode(filterImg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))  # eroding the image
    #cv2.imshow('eroding', filterImg)
    filterImg = cv2.dilate(filterImg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # dilating the image
    #cv2.imshow('dilating', filterImg)
    # convert to three channels
    mask_rbg = cv2.cvtColor(filterImg, cv2.COLOR_GRAY2BGR)
    grey = cv2.cvtColor(mask_rbg, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (25, 25), 0)
    #cv2.imshow('mask', blurred)

    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('binary', thresh1)
    thresh2 = thresh1.copy()

    contours, hierarchy = cv2.findContours((thresh1.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    if len(contours):
        cnt = max(contours, key=cv2.contourArea)

        cv2.drawContours(img, cnt, -1, (0, 255, 255), 2)

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 0)

        # Contour center
        M = cv2.moments(cnt)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx = int(cx)
        cy = int(cy)
        center = (cx, cy)
        if count == 0:
            cv2.putText(img, "Press 'Y' to set the reference position", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('y'):
                ref_position_x = cx
                ref_position_y = cy
                count = 1
            elif key & 0xFF == ord('q'):
                break
        else:
            keyboard.release(key_direction[last_index_key_direction])
            print ref_position_x, ref_position_y, cx, cy
            if cx - ref_position_x > 25:
                index_key_direction = 2
                keyboard.press(key_direction[index_key_direction])
                cv2.putText(img, key_direction[index_key_direction], (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            elif cx - ref_position_x < -25:
                index_key_direction = 3
                keyboard.press(key_direction[index_key_direction])
                cv2.putText(img, key_direction[index_key_direction], (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            elif cy - ref_position_y > 25:
                index_key_direction = 1
                keyboard.press(key_direction[index_key_direction])
                cv2.putText(img, key_direction[index_key_direction], (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            elif cy - ref_position_y < -25:
                index_key_direction = 0
                keyboard.press(key_direction[index_key_direction])
                cv2.putText(img, key_direction[index_key_direction], (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            else:
                keyboard.release(key_direction[last_index_key_direction])
                index_key_direction = last_index_key_direction
            last_index_key_direction = index_key_direction
            last_x = cx
            last_y = cy
            radius = int(radius)
            cv2.rectangle(img, (cx - radius, cy - radius), (cx + radius, cy + radius), (0, 0, 255), 0)
            crop_img = thresh2[cy - radius:cy + radius, cx - radius:cx + radius]
            # cv2.imshow('crop_img', crop_img)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
            # print(cX, cY)
            # dist = int(cv2.pointPolygonTest(cnt, (cX, cY), True))
            # cv2.circle(img, (cX, cY), dist, [0, 0, 255], 2)

            # finding convex hull
            hull = cv2.convexHull(cnt, returnPoints=False)

            # finding convexity defects
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                count_defects = 0

                # applying Cosine Rule to find angle for all defects (between fingers)
                # with angle > 90 degrees and ignore defects
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]

                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    # find length of all sides of triangle
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                    # apply cosine rule here
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                    # ignore angles > 90
                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(img, far, 5, [255, 0, 0], -1)

                # define actions required
                keyboard.release(key_function[1])
                if count_defects < 4:
                    pix_array = convert_pix(crop_img)
                    result = classify(pix_array, np_array, labels, 10)
                    for i in range(8):
                        if last_result != 'lighta':
                            keyboard.release(key_function[0])
                        if result == 'lighta':
                            cv2.putText(img, "Jump", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            keyboard.press(key_function[0])
                    last_result = result
                else:
                    cv2.putText(img, "Fire", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    keyboard.press(key_function[1])

        cv2.imshow('Gesture', img)

    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break

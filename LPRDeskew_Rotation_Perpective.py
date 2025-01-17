import datetime
#import time
import cv2
import numpy as np
import pytesseract as pt
#import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import re
#import pandas as pd


# INPUT WIDTH AND HEIGHT
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# LOAD YOLO TRAINED MODEL
net = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def get_detections(img, net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape
    
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image
    
    # GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections


def non_maximum_supression(input_image, detections):
    # FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []
    
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT
    
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5]  # probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                
                confidences.append(confidence)
                boxes.append(box)
    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # NMS
    index = np.array(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)).flatten()
    return boxes_np, confidences_np, index


def drawings(image, boxes_np, confidences_np, index):
    count=0
    # drawings
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(image, boxes_np[ind])

        # print(license_text)
        # print(conf_text)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 30), (0, 0, 0), -1)
        
        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
    return image


# predictions
def yolo_predictions(img, net):
    ## step-1: detections
    input_image, detections = get_detections(img, net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    ## step-3: Drawings
    result_img = drawings(img, boxes_np, confidences_np, index)
    return result_img



def findScore(img, angle):
    data = rotate(img, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def skewCorrect(img):
    try:
        img = cv2.resize(img, (0, 0), fx=0.85, fy=0.85)  # O 0.75 fx and fy
    except:
        pass
    delta = 1
    limit = 45
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = findScore(img, angle)
        scores.append(score)
    bestScore = max(scores)
    bestAngle = angles[scores.index(bestScore)]
    rotated = rotate(img, bestAngle, reshape=False, order=0)
    #print("[INFO] angle: {:.3f}".format(bestAngle))
    return rotated


def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)
    
    # Step 2: Move coordinate system to centre of object
    shifted = pts - center
    
    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])
    
    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]


def getContours(img, orig):  # Change - pass the original image too
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i
        
        warped = None  # Stores the warped license plate image
        if index is not None:  # Draw the biggest contour on the image
            cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)
            
            src = np.squeeze(biggest).astype(np.float32)  # Source points
            height = skewCorrect(img).shape[0]
            width = skewCorrect(img).shape[1]
            # Destination points
            dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])
            
            # Order the points correctly
            biggest = order_points(src)
            dst = order_points(dst)
            
            # Get the perspective transform
            M = cv2.getPerspectiveTransform(src, dst)
            
            # Warp the image
            
            img_shape = (width, height)
            warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)
            # j = 0
            # plt.imsave("red_quadrilateral_warped" + str(j) + ".jpg", warped)
            # warped = plt.imread("red_quadrilateral_warped" + str(j) + ".jpg")
            # j += 1
    
        return biggest, imgContour, warped
count = 0
def extract_text(image, bbox):
    global count
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]
    # try:
    #     cv2.imshow('ROI', roi)
    #
    # except:
    #     pass
    roi = skewCorrect(roi)
    kernel = np.ones((4, 3), np.uint8)
    # kernel = np.array([[0, -1, 0],
    #                [-1,10, -1],
    #                [0, -1, 0]])
    try:
        grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(grayscale, 11, 17, 17)
        imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 20, 20)  # 150, 200
        imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
        imgThres = cv2.erode(imgDial, kernel, iterations=2)
        biggest, imgContour, warped = getContours(imgThres, roi)
    except:
        pass
    if 0 in roi.shape:
        return ''
    
    elif warped is not None:
        text = pt.image_to_string(warped, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6")
        text = text.strip()
        text = text.replace('\n', '').replace('¥','').replace(' ', '').replace('~', '').replace('\\', '').replace('.', '').replace('/', '').replace("’", '').replace(',', '').replace("'", '').replace('-', '').replace('>', '').replace('<', '').replace('$', '').replace('%', '').replace('|', '').replace('@', '').replace('!', '').replace('¡', '').replace('°', '').replace('&', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace(':', '').replace(';', '').replace('‘','').replace('?','').replace('¿','').replace('(','').replace(')','').replace('#','').replace('* ','').replace('+','')
        text = text.upper()
        #patron = '[A-Z][0-9][0-9][0-9][A-Z][A-Z]||[A-Z][A-Z][0-9][0-9][A-Z][A-Z]'
        patron = '[A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9]'
        if re.match(patron,text) and len(text) ==7:
            cv2.imwrite("Resources/NumberPlate/NoPlate_" + str(count) + ".jpg", roi)
            count += 1
            print(text, datetime.datetime.now())
        return text
    elif 0 not in roi.shape:
        text = pt.image_to_string(roi, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6")
        text = text.strip()
        text = (text.replace('\n', '').replace('¥','').replace(' ', '').replace('~', '').replace('\\', '').replace('.', '').replace('/', '').replace("’", '').replace(',', '').
                replace("'", '').replace('-', '').replace('>', '').replace('<', '').replace('$', '').replace('%', '').replace('|', '').replace('@', '').replace('!', '').replace('¡', '').
                replace('°', '').replace('&', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace(':', '').replace(';', '').replace('‘','').replace('?','').
                replace('¿','').replace('(','').replace(')','').replace('#','').replace('* ','').replace('+',''))
        text = text.upper()
        #patron = '[A-Z][A-Z]+[0-9][0-9]+[A-Z][A-Z]||[A-Z][A-Z][0-9][0-9][A-Z][A-Z]'
        patron = '[A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9]'
        if re.match(patron,text) and len(text) == 7:
            cv2.imwrite("Resources/NumberPlate/NoPlate_" + str(count) + ".jpg", roi)
            count += 1
            print(text, datetime.datetime.now())
        return text

#patron = re.compile('[A-Z]{3}-[0-9]{4}]')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret == False:
        print('unable to read')
        break
    
    results = yolo_predictions(frame, net)
    cv2.namedWindow('VISOR', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('VISOR', results)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()


'''
            cv2.imwrite("../Resources/NumberPlate/NoPlate_"+str(count)+".jpg", frameROI)
            cv2.rectangle(frame, (0,200), (640, 300), (0,255,0), cv2.FILLED)
            cv2.putText(frame, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 2)
            cv2.imshow("Output Video", frame)
            cv2.waitKey(500)
            count+=1


'''
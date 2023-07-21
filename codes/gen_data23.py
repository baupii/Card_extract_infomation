import cv2
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import imutils
import glob
import random
import csv
from PIL import Image, ImageOps
import os
import time

def rotate(image, angle, center = None, scale = 1):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.resize(warped, (1000, 600))
    return tl, tr, br, bl, warped

def order_points2(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform2(image, pts):

    rect = order_points2(pts)
    (tl, tr, br, bl) = rect
    return tl, tr, br, bl

def preprocess2(img):
    img = img.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.dilate(gray,kernel,iterations = 1)
    gau = cv2.GaussianBlur(erosion, (5, 5), 0)
    edged = cv2.Canny(gau, 3, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
#             print(type(screenCnt))
#             print(screenCnt)
            break
        else:
            screenCnt = np.array([[0,0], [0, 0], [0,0], [0, 0]])
            

    tl, tr, br, bl = four_point_transform2(img.copy(), screenCnt.reshape(4, 2))
    return tl, tr, br, bl, edged

def add_white_border(image, border_size):

    # Calculate the new dimensions
    height, width, _ = image.shape
    new_width = width + (2 * border_size)
    new_height = height + (2 * border_size)

    # Create a new white background image
    bordered_image = cv2.copyMakeBorder(image, border_size, border_size, 
                                                border_size, border_size, 
                                                cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return bordered_image


# load all image in folder in a list

# /home/lam/bau/CCCD/detect4point/dataset
bright_image = []
# for image in glob.glob('C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/imgs/bright_image/*'):
for image in glob.glob('/home/lam/bau/CCCD/detect4point/dataset/bright_image/*'):
    img = cv2.imread(image)
    bright_image.append(img)
    
print(len(bright_image))
def add_bright(image):
    bright = random.choice(bright_image)
    flipp = [-1, 0, 1]
    fl = random.choice(flipp)
    bright = cv2.flip(bright, fl)
    bright = cv2.resize(bright, (image.shape[1], image.shape[0]))
    alpha = 0.6
    beta = 1 - alpha
    gamma = 0
    out = cv2.addWeighted(image, alpha, bright, beta, gamma)
    return out

def add_border_black(image):
    height, width, _ = image.shape
    border = cv2.copyMakeBorder(
        image,
        top=int(0.06*height),
        bottom=int(0.06*height),
        left=int(0.06*width),
        right=int(0.06*width),
        borderType=cv2.BORDER_CONSTANT,
        value=[0,0,0]
    )
    return border

def add_border_white2(image):
    height, width, _ = image.shape
    border = cv2.copyMakeBorder(
        image,
        top=int(0.06*height),
        bottom=int(0.06*height),
        left=int(0.06*width),
        right=int(0.06*width),
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    return border

# -----------------main-----------------

# path_bg = "C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/imgs/bg_all/*"
# path_img = "C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/imgs/card/crop_ID_card_real/*"
path_bg = "/home/lam/bau/CCCD/detect4point/dataset/bg_all/*"
path_img = "/home/lam/bau/CCCD/detect4point/dataset/crop_ID_card_real/*"


a = 0


# path_save_img_train = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/imgs/output4/images/train/'
# path_save_label_train = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/imgs/output4/labels/train/'

# path_save_img_val = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/imgs/output4/images/val/'
# path_save_label_val = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/imgs/output4/labels/val/'
path_save_img_train = '/home/lam/bau/CCCD/detect4point/yolo/dataset_new/images/train/'
path_save_label_train = '/home/lam/bau/CCCD/detect4point/yolo/dataset_new/labels/train/'

path_save_img_val = '/home/lam/bau/CCCD/detect4point/yolo/dataset_new/images/val/'
path_save_label_val = '/home/lam/bau/CCCD/detect4point/yolo/dataset_new/labels/val/'



for image in glob.glob(path_img):
    
    for file in glob.glob(path_bg):
        print(file)
        flipp = [-1, 0, 1]
        fl = random.choice(flipp)
        bg = cv2.imread(file)
        while bg.shape[0] < 500 or bg.shape[1] < 500:
            bg = cv2.resize(bg, (bg.shape[1]*2, bg.shape[0]*2))
        bg = cv2.flip(bg, fl)
        h, w, _ = bg.shape
        black_bg = np.zeros((h, w, 3))

        img = cv2.imread(image)
        h_, w_, _ = img.shape

#        add border
        border_size = 10
        img_border = add_white_border(img, border_size)

        #resize card
        width_rand_ratio = np.random.uniform(0.85, 0.95)
        height_rand_ratio = np.random.uniform(0.85, 0.95)
        if w < h:
            w_new = int(w*width_rand_ratio)
            h_new = int(h_*(w_new/w_))
        else:
            h_new = int(h*height_rand_ratio)
            w_new = int(w_*(h_new/h_))

        out_auto = cv2.resize(img, (w_new, h_new) )
        out_auto2 = cv2.resize(img_border, (w_new, h_new) )
        h_, w_, _ = out_auto.shape


        x_min = int(w*0.03)
        y_min = int(h*0.03)
        x_max = int(w*0.05)
        y_max = int(h - h_new - 10)

        x1 = np.random.randint(x_min, x_max)
        y1 = np.random.randint(y_min, y_max)

        # set time 
        # start_time = time.time()
        # check = True
        if (y1+h_) + 20 >= h or (x1+w_) + 20>= w:
            # x1 = np.random.randint(x_min, x_max)
            # y1 = np.random.randint(y_min, y_max)
            continue
            # if time.time() - start_time > 0.05:
            #     check = False
            #     break
        # if check == False:
        #     continue
        bg[y1:y1+h_ , x1:x1+w_] = out_auto
        black_bg[y1:y1+h_ , x1:x1+w_] = out_auto2

        # rotate img
        angle = np.random.randint(-1,1)
        out_bg = rotate(bg, angle)
        h,w,_ = out_bg.shape
        out_black_bg = rotate(black_bg, angle)


        if a < 5000:
            
            h, w, _ = bg.shape
            try:
                out2 = add_border_black(black_bg)
                tl, tr, br, bl, canny = preprocess2(out2)
            except:
                # print('error')
                continue

            fi = str(a)
            # if not os.path.exists('train'):
            #     os.makedirs('train')
            path_save1 = path_save_img_train + fi + "_6.jpg"
            file_path = path_save_label_train + fi + '_6.txt'
            print(path_save1)
            if a % 10 == 0:
                bg = add_bright(bg)
            out1 = add_border_white2(bg)
            cv2.imwrite(path_save1, out1)
            

        


        #val

        elif a >= 5000 and a < 7500:

            h, w, _ = bg.shape
            try:
                out2 = add_border_black(black_bg)
                tl, tr, br, bl, canny = preprocess2(out2)
            except:
                # print('error')
                continue

            fi = str(a)
            path_save1 = path_save_img_val + fi + "_val2.jpg"
            file_path = path_save_label_val + fi + '_val2.txt'
            print(path_save1)
            if a % 10 == 0:
                bg = add_bright(bg)
            out1 = add_border_white2(bg)
            cv2.imwrite(path_save1, out1)
        
        h, w, _ = out2.shape
        tl, tr, br, bl, canny = preprocess2(out2)
        with open(file_path, 'w') as file:
            # Append some text to the file
            file.write("0 "+ str(tl[0]/w) +" " + str(tl[1]/h) +" "+str(0.05)+" "+ str(0.05) +"\n")
            file.write("1 "+ str(tr[0]/w) +" " + str(tr[1]/h) +" "+str(0.05)+" " +str(0.05) +"\n")
            file.write("2 "+ str(br[0]/w) +" " + str(br[1]/h) +" "+str(0.05)+" " +str(0.05) +"\n")
            file.write("3 "+ str(bl[0]/w) +" " + str(bl[1]/h) +" "+str(0.05)+" " +str(0.05))
        a+=1
        print(a)
        if a == 7500:
            break
    if a == 7500: break   
        

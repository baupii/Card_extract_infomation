import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imutils
import glob
import random
import csv
from PIL import Image, ImageOps

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

def trans1(out_bg, out_black_bg):
#     flexible = np.random.uniform(0.98,1.02)
    input_pts = np.float32([[w*0.1, h*0.1],[w, 0],[w, h],[w*0.1, h]])
    output_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out1 = cv2.warpPerspective(out_bg.copy(),M,(w, h),flags=cv2.INTER_LINEAR)
    out2 = cv2.warpPerspective(out_black_bg.copy(),M,(w, h),flags=cv2.INTER_LINEAR)
    
    return out1, out2

def trans2(out_bg, out_black_bg):
#     flexible = np.random.uniform(0.98,1.02)
    input_pts = np.float32([[0, 0],[w*0.9, h*0.1],[w*0.9, h],[0, h]])
    output_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out1 = cv2.warpPerspective(out_bg.copy(),M,(w, h),flags=cv2.INTER_LINEAR)
    out2 = cv2.warpPerspective(out_black_bg.copy(),M,(w, h),flags=cv2.INTER_LINEAR)
    
    return out1, out2

def trans3(out_bg, out_black_bg):
#     flexible = np.random.uniform(0.98,1.02)
    input_pts = np.float32([[w*0.1, 0],[w, 0],[w, h],[w*0.1, h*0.9]])
    output_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out1 = cv2.warpPerspective(out_bg.copy(),M,(w, h),flags=cv2.INTER_LINEAR)
    out2 = cv2.warpPerspective(out_black_bg.copy(),M,(w, h),flags=cv2.INTER_LINEAR)
    return out1, out2

    
def trans4(out_bg, out_black_bg):
#     flexible = np.random.uniform(0.98,1.02)
    input_pts = np.float32([[0, 0],[w*0.9, 0],[w*0.9, h*0.9],[0, h]])
    output_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out1 = cv2.warpPerspective(out_bg.copy(),M,(w, h),flags=cv2.INTER_LINEAR)
    out2 = cv2.warpPerspective(out_black_bg.copy(),M,(w, h),flags=cv2.INTER_LINEAR)
    return out1, out2

path_bg = "/home/lam/bau/CCCD/detect4point/dataset/bg/*"
path_img = "/home/lam/bau/CCCD/detect4point/dataset/crop_ID_card_real/*"
a = 0

#     print(h, w)\
for image in glob.glob(path_img):
    
    for file in glob.glob(path_bg):
        print(file)
        flipp = [-1, 0, 1]
        fl = random.choice(flipp)
        bg = cv2.imread(file)
        bg = cv2.flip(bg, fl)
        h, w, _ = bg.shape
        black_bg = np.zeros((h, w, 3))

        img = cv2.imread(image)
        h_, w_, _ = img.shape
#        add border
        border_size = 10
        img_border = add_white_border(img, border_size)

        
        #resize card
        w_new = int(w*0.35)
        h_new = int(h_*(w_new/w_))

        out_auto = cv2.resize(img, (w_new, h_new) )
        out_auto2 = cv2.resize(img_border, (w_new, h_new) )
        h_, w_, _ = out_auto.shape


        x_min = int(w*0.2)
        y_min = int(h*0.2)
        x_max = int(w*0.25)
        y_max = int(h*0.25)

        x1 = np.random.randint(x_min, x_max)
        y1 = np.random.randint(y_min, y_max)


        #add card in bg
        if  (y1+h_) < h:
            bg[y1:y1+h_ , x1:x1+w_] = out_auto
            black_bg[y1:y1+h_ , x1:x1+w_] = out_auto2
        
        else:
            continue

        # rotate img
        angle = np.random.randint(-3,3)
        out_bg = rotate(bg, angle)
        h,w,_ = out_bg.shape
        out_black_bg = rotate(black_bg, angle)

        # Tranform img (làm méo hình)
        if a < 2500:
            out1, out2 = trans1(out_bg, out_black_bg)
            h, w, _ = out1.shape

            tl, tr, br, bl, canny = preprocess2(out2)
            min_x = min(tl[0], bl[0])
            max_x = max(tr[0], br[0])

            ab = np.random.uniform(0.6, 0.9)
            bc = np.random.uniform(0.1, 0.4)
            out_new1 = out1[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]
            out_new2 = out2[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]

            fi = str(a)
            path_save1 = "/home/lam/bau/CCCD/detect4point/yolo/dataset1/images/train/" + fi + "_1.jpg"
            file_path = '/home/lam/bau/CCCD/detect4point/yolo/dataset1/labels/train/' + fi + '_1.txt'
            cv2.imwrite(path_save1, out_new1)
            
        elif a >= 2500 and a < 5000:
            out1, out2 = trans2(out_bg, out_black_bg)
            h, w, _ = out1.shape

            tl, tr, br, bl, canny = preprocess2(out2)
            min_x = min(tl[0], bl[0])
            max_x = max(tr[0], br[0])

            ab = np.random.uniform(0.6, 0.9)
            bc = np.random.uniform(0.1, 0.4)
            out_new1 = out1[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]
            out_new2 = out2[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]

            fi = str(a)
            path_save1 = "/home/lam/bau/CCCD/detect4point/yolo/dataset1/images/train/" + fi + "_2.jpg"
            file_path = '/home/lam/bau/CCCD/detect4point/yolo/dataset1/labels/train/' + fi + '_2.txt'
            cv2.imwrite(path_save1, out_new1)
            
        elif a >= 5000 and a < 7500:
            out1, out2 = trans3(out_bg, out_black_bg)
            h, w, _ = out1.shape

            tl, tr, br, bl, canny = preprocess2(out2)
            min_x = min(tl[0], bl[0])
            max_x = max(tr[0], br[0])

            ab = np.random.uniform(0.6, 0.9)
            bc = np.random.uniform(0.1, 0.4)
            out_new1 = out1[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]
            out_new2 = out2[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]

            fi = str(a)
            path_save1 = "/home/lam/bau/CCCD/detect4point/yolo/dataset1/images/train/" + fi + "_3.jpg"
            file_path = '/home/lam/bau/CCCD/detect4point/yolo/dataset1/labels/train/' + fi + '_3.txt'
            cv2.imwrite(path_save1, out_new1)
            
        elif a >= 7500 and a < 1000:
            out1, out2 = trans3(out_bg, out_black_bg)
            h, w, _ = out1.shape

            tl, tr, br, bl, canny = preprocess2(out2)
            min_x = min(tl[0], bl[0])
            max_x = max(tr[0], br[0])

            ab = np.random.uniform(0.6, 0.9)
            bc = np.random.uniform(0.1, 0.4)
            out_new1 = out1[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]
            out_new2 = out2[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]

            fi = str(a)
            path_save1 = "/home/lam/bau/CCCD/detect4point/yolo/dataset1/images/train/" + fi + "_4.jpg"
            file_path = '/home/lam/bau/CCCD/detect4point/yolo/dataset1/labels/train/' + fi + '_4.txt'
            cv2.imwrite(path_save1, out_new1)
            
        else:
            out1, out2 = trans4(out_bg, out_black_bg)
            h, w, _ = out1.shape

            tl, tr, br, bl, canny = preprocess2(out2)
            min_x = min(tl[0], bl[0])
            max_x = max(tr[0], br[0])

            ab = np.random.uniform(0.6, 0.9)
            bc = np.random.uniform(0.1, 0.4)
            out_new1 = out1[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]
            out_new2 = out2[0:h , int(min_x * ab): int(max_x + (w- max_x)* bc)]

            fi = str(a)
            path_save1 = "/home/lam/bau/CCCD/detect4point/yolo/dataset1/images/val/" + fi + ".jpg"
            file_path = '/home/lam/bau/CCCD/detect4point/yolo/dataset1/labels/val/' + fi + '.txt'
            cv2.imwrite(path_save1, out_new1)
        
        h, w, _ = out_new2.shape
        tl, tr, br, bl, canny = preprocess2(out_new2)
        with open(file_path, 'w') as file:
            # Append some text to the file
            file.write("0 "+ str(tl[0]/w) +" " + str(tl[1]/h) +" "+str(0.05)+" "+ str(0.05) +"\n")
            file.write("1 "+ str(tr[0]/w) +" " + str(tr[1]/h) +" "+str(0.05)+" " +str(0.05) +"\n")
            file.write("2 "+ str(br[0]/w) +" " + str(br[1]/h) +" "+str(0.05)+" " +str(0.05) +"\n")
            file.write("3 "+ str(bl[0]/w) +" " + str(bl[1]/h) +" "+str(0.05)+" " +str(0.05))
        a+=1
        print(a)
        if a == 12000:
            break
    if a == 12000: break   
        
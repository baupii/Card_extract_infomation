import cv2
import numpy as np
import glob
from pandas import read_csv


# Create a function based on a CV2 Event (Left button click)
ix,iy = -1,-1
BOX = []
RESIZE_RATIO = 2

# mouse callback function
def draw_point(event,x,y,flags,param):
    global ix, iy, DRAWING, OBJECT_IMAGE, BOX
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Then we take note of where that mouse was located
        ix,iy = x,y
        BOX.append('%6.5f, %6.5f' % (x / w, y / h))

    elif event == cv2.EVENT_LBUTTONUP:
        # we complete the rectangle.
        cv2.circle(img, (x,y), 5, (0,255,0), -1)


# Create a black image
name_list = glob.glob('C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/imgs/truoc/*')
try:
    df = read_csv('C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/codes/data_edit1.csv')
    existed_file = df.iloc[:, 0].values

    f = open('C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/codes/data_edit1.csv', 'a')
except:
    f = open('C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/codes/data_edit1.csv', 'w')
    existed_file = []

for file_name in name_list:
    if file_name in existed_file:
        continue

    img = cv2.imread(file_name)
    img = cv2.resize(img, (750 , 750 ))
    h, w = img.shape[:2]
    img0 = img.copy()

    # This names the window so we can reference it 
    cv2.namedWindow(winname='my_drawing')
    # Connects the mouse button to our callback function
    cv2.setMouseCallback('my_drawing',draw_point)

    cout = 0
    while True: #Runs forever until we break with Esc key on keyboard
        # Shows the image window
        cv2.imshow('my_drawing',img)
        
        # Check if ESC, 's', or 'c' key was pressed
        k = cv2.waitKey(1) 
        esc_k = k & 0xFF == 27
        save_defect = k & 0xFF == ord('s')
        clear = k & 0xFF == ord('c')
        no_plate = k & 0xFF == ord('n')
        if esc_k:
            break
        # Press 's' key to select points
        if save_defect:
            if len(BOX) == 4:
                cout += 1
                line = ', '.join(BOX)
                print(line)
                f.write('%s, %s, 1 \n' % (file_name, line))
                BOX = []
                break
            else:
                print('Please select more points')

        # press 'c' key to clean the selections
        if clear:
            img = img0.copy()
            BOX = []

        # press 'n' if no plate is found
        if no_plate:
            line = ','.join(["%6d" % 0] * 8)
            print(line)
            f.write('%s, %s, 0 \n' % (file_name, line))
            BOX = []
            break
    #Press ESC to exit
    if esc_k:
        break
cv2.destroyAllWindows()
f.close()
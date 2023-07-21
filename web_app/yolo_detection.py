# config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = '/home/lampt/bau/web_app/model/transformerocr.pth'
# config['device'] = 'cpu' # device chạy 'cuda:0', 'cuda:1', 'cpu'
# detector = Predictor(config)

# path_a = '/home/lampt/bau/web_app/static/a/'
# path_b = '/home/lampt/bau/web_app/static/b/'
# path_c = '/home/lampt/bau/web_app/static/c/'
# path_d = '/home/lampt/bau/web_app/static/d/'

#  app.run(debug = False, port=6006, host='0.0.0.0')

import torch
import cv2
import numpy as np
# from ultralytics import YOLO
# path_model  = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/model/'
path_model  = '/home/lampt/bau/web_app/model/'
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path = path_model + 'last3.pt')
model1.conf = 0.2
model_star = torch.hub.load('ultralytics/yolov5', 'custom', path = path_model + 'last5.pt')
model_star.conf = 0.2

model2 = torch.hub.load('ultralytics/yolov5', 'custom', path = path_model + 'last4.pt')
model2.conf = 0.1

def detected_star(img_path):
    tl = []
    tr = []
    br = []
    bl = []

    # mess = ''
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # orig = img.copy()

    border_size = 150
    img = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[125,125,125]
        )
    orig = img.copy()
    rs = model_star(img)
    out = rs.pandas().xyxy[0]

    if len(out[out['name']== 'tl']) == 1 and len(out[out['name']== 'tr']) == 1 and len(out[out['name']== 'br']) == 1 and len(out[out['name']== 'bl']) == 1:
        for i in range(len(out)):
        #     print(out['name'][i])
            if out['name'][i] == 'br':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                br.append(x)
                br.append(y)

            elif out['name'][i] == 'tr':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                tr.append(x)
                tr.append(y)

            elif out['name'][i] == 'tl':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                tl.append(x)
                tl.append(y)

            else:
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                bl.append(x)
                bl.append(y)

        input_pts = np.float32([tl, tr, br, bl])

        output_pts = np.float32([[0, 0],
                                [1000, 0],
                                [1000, 600],
                                [0, 600]])

        M = cv2.getPerspectiveTransform(input_pts,output_pts)

        warped = cv2.warpPerspective(img,M,(1000, 600),flags=cv2.INTER_LINEAR)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        mess = 'oke'        

    elif len(out[out['name']== 'tl']) == 2 and len(out[out['name']== 'tr']) == 2 and len(out[out['name']== 'br']) == 2 and len(out[out['name']== 'bl']) == 2:
        mess = 'Chụp 1 hình thui'
        warped = None
        orig = None
    else:
        mess = 'Không nhận diện được'
        warped = None
        orig = None

    return tl, tr, br, bl, warped, orig, mess  

def detected(img_path):
    tl = []
    tr = []
    br = []
    bl = []

    # mess = ''
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # orig = img.copy()

    border_size = 150
    img = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[125,125,125]
        )
    orig = img.copy()
    rs = model1(img)
    out = rs.pandas().xyxy[0]

    if len(out[out['name']== 'tl']) == 1 and len(out[out['name']== 'tr']) == 1 and len(out[out['name']== 'br']) == 1 and len(out[out['name']== 'bl']) == 1:
        for i in range(len(out)):
        #     print(out['name'][i])
            if out['name'][i] == 'br':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                br.append(x)
                br.append(y)

            elif out['name'][i] == 'tr':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                tr.append(x)
                tr.append(y)

            elif out['name'][i] == 'tl':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                tl.append(x)
                tl.append(y)

            else:
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                bl.append(x)
                bl.append(y)

        input_pts = np.float32([tl, tr, br, bl])

        output_pts = np.float32([[0, 0],
                                [1000, 0],
                                [1000, 600],
                                [0, 600]])

        M = cv2.getPerspectiveTransform(input_pts,output_pts)

        warped = cv2.warpPerspective(img,M,(1000, 600),flags=cv2.INTER_LINEAR)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        mess = 'oke'        

    elif len(out[out['name']== 'tl']) == 2 and len(out[out['name']== 'tr']) == 2 and len(out[out['name']== 'br']) == 2 and len(out[out['name']== 'bl']) == 2:
        mess = 'Chụp 1 hình thui'
        warped = None
        orig = None
    else:
        mess = 'Không nhận diện được'
        warped = None
        orig = None

    return tl, tr, br, bl, warped, orig, mess    


def detected2(img_path):
    tl = []
    tr = []
    br = []
    bl = []

    # mess = ''
    
    img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.copy()
    # img2 = cv2.bilateralFilter(img2,9,75,75)
    # orig = img.copy()

    border_size = 350
    img = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    orig = img.copy()
    rs = model2(img)
    out = rs.pandas().xyxy[0]

    if len(out[out['name']== 'tl']) == 1 and len(out[out['name']== 'tr']) == 1 and len(out[out['name']== 'br']) == 1 and len(out[out['name']== 'bl']) == 1:
        for i in range(len(out)):
        #     print(out['name'][i])
            if out['name'][i] == 'br':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                br.append(x)
                br.append(y)

            elif out['name'][i] == 'tr':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                tr.append(x)
                tr.append(y)

            elif out['name'][i] == 'tl':
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                tl.append(x)
                tl.append(y)

            else:
                x = int((out['xmin'][i] + out['xmax'][i]) // 2)
                y = int((out['ymin'][i] + out['ymax'][i]) // 2)
                bl.append(x)
                bl.append(y)

        input_pts = np.float32([tl, tr, br, bl])

        output_pts = np.float32([[0, 0],
                                [1000, 0],
                                [1000, 600],
                                [0, 600]])

        M = cv2.getPerspectiveTransform(input_pts,output_pts)

        warped = cv2.warpPerspective(img,M,(1000, 600),flags=cv2.INTER_LINEAR)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        mess = 'oke'        

    elif len(out[out['name']== 'tl']) == 2 and len(out[out['name']== 'tr']) == 2 and len(out[out['name']== 'br']) == 2 and len(out[out['name']== 'bl']) == 2:
        mess = 'Chụp 1 hình thui'
        warped = None
        orig = None
    else:
        mess = 'Không nhận diện được'
        warped = None
        orig = None

    return tl, tr, br, bl, warped, orig, mess 



def plot(img):
        tl, tr, br, bl, warped, orig, mess = detected(img)
        img_new = cv2.circle(orig, [int(t) for t in tl], 20, (0,255,0), -1)
        img_new = cv2.circle(orig, [int(t) for t in tr], 20, (0,255,0), -1)
        img_new = cv2.circle(orig, [int(t) for t in bl], 20, (0,255,0), -1)
        img_new = cv2.circle(orig, [int(t) for t in br], 20, (0,255,0), -1)
        abc = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
        return abc



#cái này của mặt trước cccd mới
def information(warped):
    name_ = warped[323:374, 290:800]
    ID_ = warped[240:300, 400:760]
    date_ = warped[365:405, 560:760]
    sex_ = warped[390:460, 460:570]
    origin_ = warped[470:520, 290:900]
    residence1_ = warped[505:550, 685:900]
    residence2_ = warped[545:620, 290:900]
    return name_, ID_, date_, sex_, origin_, residence1_, residence2_



#cái này của mặt sau CCCD mới
def information2(warped):
    datee = warped[70:115 , 405:545]
    return datee



#cái này của mặt trước cccd cũ
def information3(warped):
    warped = cv2.resize(warped, (1000, 600))
    ID_ = warped[150:250, 430:1000]
    name_ = warped[210: 320, 400: 1000]
    date_ = warped[290: 370, 550: 1000]
    sex_ = warped[340: 400, 400: 550]
    # origin_ = warped[400: 510, 450: 1000]
    origin1_ = warped[410:450, 440:1000]
    origin2_ = warped[440:495, 440:1000]
    residence1_ = warped[465: 535, 480: 1000]
    residence2_ = warped[510: 770, 410: 1000]

    return name_, ID_, date_, sex_, origin1_, origin2_, residence1_, residence2_



#cái này là của mặt trước CMND
def information4(warped):
    # warped = cv2.resize(warped, (1000, 600))
    ID_ = warped[130: 210, 460: 950]
    name1_ = warped[180:270, 400:950]
    name2_ = warped[250:320, 280:950] 
    date_ = warped[300:380, 440:950]
    origin1_ = warped[370:430, 500:950]
    origin2_ = warped[410:500, 280:950]
    residence1_ = warped[470:540, 590:950]
    residence2_ = warped[520:590, 280:950]

    return name1_, name2_, ID_, date_, origin1_, origin2_, residence1_, residence2_




#cái này là ủa của mặt sau CCCD
def information7(warped):
    d1 = warped[280:330, 500:980]

    # d2 = warped[280:330, 750:800]

    # d3 = warped[280:330, 870:1000]
    return d1



#cái này là của mặt sau CMND
def information6(warped):
    d1 = warped[300:360, 400:950]
    # d2 = warped[300:350, 670:760]
    # d3 = warped[300:350, 820:970]
    return d1 










def resize(img): 
    w, h = img.shape
    if (32/w < 416/h):
        scale_percent = 32/w
    else:
        scale_percent = 416/h
    height = int(img.shape[0] * scale_percent)
    width = int(img.shape[1] * scale_percent)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return img
    
def reshape_expand_dim(img):
    w, h = img.shape
    if w < 32:
            add_zeros = np.ones((32-w, h))*210
            img = np.concatenate((img, add_zeros))
    if h < 416:
        add_zeros = np.ones((32, 416-h))*210
        img = np.concatenate((img, add_zeros), axis=1)
    img = np.expand_dims(img , axis = 2)

    return img

def read_data(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    img = resize(img)
    img = reshape_expand_dim(img)
    img = img/255.
    img = np.array(img)

    return img
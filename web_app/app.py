from flask import Flask, render_template, request, jsonify
# from load_model import model
# from load_model import predict
# from padded_img import resize_with_padding
import cv2
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import pyheif
import PIL.Image
from flask import Flask, render_template, request, send_from_directory
from yolo_detection import detected_star, detected, detected2, plot, resize, reshape_expand_dim, read_data, information, information2, information3, information4, information6, information7 

from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/model/transformerocr.pth'
# config['device'] = 'cpu' # device chạy 'cuda:0', 'cuda:1', 'cpu'
# detector = Predictor(config)

# path_a = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/a/'
# path_b = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/b/'
# path_c = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/'
# path_d = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/d/'


config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = '/home/lampt/bau/web_app/model/transformerocr.pth'
config['device'] = 'cpu' # device chạy 'cuda:0', 'cuda:1', 'cpu'
detector = Predictor(config)

path_a = '/home/lampt/bau/web_app/static/a/'
path_b = '/home/lampt/bau/web_app/static/b/'
path_c = '/home/lampt/bau/web_app/static/c/'
path_d = '/home/lampt/bau/web_app/static/d/'

def check_number(string):
    for i in string:
        if i.isdigit():
            return True
    return False

def convert_number(a):
    list_number = []
    for i in range(len(a)):
        if a[i].isdigit():
            list_number.append(a[i])
    result = ''.join(list_number)
    try:
        result = result[:2] + '-' + result[2:4] + '-' + result[4:]
    except:
        pass
    return result



app = Flask(__name__)

@app.route("/")
def view_home():
    return render_template("index.html")

@app.route("/first")
def view_first_page():
    return render_template("viet_word.html")

@app.route("/second")
def view_second_page():
    return render_template("cccd_truoc.html")

@app.route("/third")
def view_third_page():
    return render_template("cmnd_truoc.html")


def heic_to_opencv(heic_file):
    # Read the HEIC file using pyheif
    heif_file = pyheif.read(heic_file)
    # Extract the image data and metadata
    image_data = heif_file.data
    mode = heif_file.mode
    size = heif_file.size

    # Create a PIL image from the extracted data
    pil_image = PIL.Image.frombytes(mode, size, image_data, "raw", mode, 0, 1)

    # Convert the PIL image to a numpy array
    np_array = np.array(pil_image)

    # Convert the image from RGB to BGR (OpenCV uses BGR format)
    bgr_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

    return bgr_image


# Đây là kết quả mặt trước CCCD mới
@app.route("/submit_word", methods = ['GET', 'POST'])
def main1():
    if request.method == 'POST':
        img = request.files['my_image']
        path1 = img.filename 

        name = path1.split('.')[0]
        tail = path1.split('.')[1]
        if tail ==  'heic' or tail ==  'HEIC':
            # Usage example
            heic_file_path = path_a + path1
            opencv_image = heic_to_opencv(heic_file_path)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_a + name + '_.jpg', opencv_image)
            img_path = path_a + name + '_.jpg'
            path1 = name + '_.jpg'
            print('abc',path1)
            print(img_path)

        else:
            img_path = path_a + path1
            print(img_path) 
            img.save(img_path)

        
        tl, tr, br, bl, warped, orig, mess = detected(img_path)
        if  mess == 'oke':
            
            # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
            img_path3 = path_c + path1 
            
            cv2.imwrite(img_path3, warped)

            #-------------------------------------
            # img = cv2.imread('C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1)
            img = cv2.imread(path_c + path1)

            name_, ID_, date_, sex_, origin_, residence1_, residence2_ = information(img)
            # img_path4 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/d/'
            img_path4 = path_d

            cv2.imwrite(img_path4 + 'a.jpg', name_)
            cv2.imwrite(img_path4 + 'b.jpg', ID_)
            cv2.imwrite(img_path4 + 'c.jpg', date_)
            cv2.imwrite(img_path4 + 'd.jpg', sex_)
            cv2.imwrite(img_path4 + 'e.jpg', origin_)
            cv2.imwrite(img_path4 + 'f.jpg', residence1_)
            

            name_ = Image.fromarray(name_)

            date_ = cv2.cvtColor(date_, cv2.COLOR_BGR2RGB)
            date_ = Image.fromarray(date_)

            sex_ = cv2.cvtColor(sex_, cv2.COLOR_BGR2RGB)
            sex_ = Image.fromarray(sex_)

            ID_ = cv2.cvtColor(ID_, cv2.COLOR_BGR2RGB)
            ID_ = Image.fromarray(ID_)

            origin_ = cv2.cvtColor(origin_, cv2.COLOR_BGR2RGB)
            origin_ = Image.fromarray(origin_)

            residence1_ = cv2.cvtColor(residence1_, cv2.COLOR_BGR2RGB)
            residence1_ = Image.fromarray(residence1_)

            residence2_ = cv2.cvtColor(residence2_, cv2.COLOR_BGR2RGB)
            residence2_ = Image.fromarray(residence2_)
            
            name = detector.predict(name_, return_prob=False)
            date = detector.predict(date_, return_prob=False)
            gender = detector.predict(sex_, return_prob=False)
            ID = detector.predict(ID_, return_prob=False)
            origin = detector.predict(origin_, return_prob=False)
            residence1 = detector.predict(residence1_, return_prob=False)
            residence2 = detector.predict(residence2_, return_prob=False)
            residence = residence1 + ', ' + residence2
            #-------------------------------------


            return render_template("viet_word.html", path1 = path1, name = name, ID = ID,\
                                date = date, gender = gender, origin = origin, residence = residence)
        else:
            tl, tr, br, bl, warped, orig, mess = detected_star(img_path)
            if  mess == 'oke':
            
                # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
                img_path3 = path_c + path1 
                
                cv2.imwrite(img_path3, warped)

                #-------------------------------------
                # img = cv2.imread('C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1)
                img = cv2.imread(path_c + path1)

                name_, ID_, date_, sex_, origin_, residence1_, residence2_ = information(img)
                # img_path4 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/d/'
                img_path4 = path_d

                cv2.imwrite(img_path4 + 'a.jpg', name_)
                cv2.imwrite(img_path4 + 'b.jpg', ID_)
                cv2.imwrite(img_path4 + 'c.jpg', date_)
                cv2.imwrite(img_path4 + 'd.jpg', sex_)
                cv2.imwrite(img_path4 + 'e.jpg', origin_)
                cv2.imwrite(img_path4 + 'f.jpg', residence1_)
                

                name_ = Image.fromarray(name_)

                date_ = cv2.cvtColor(date_, cv2.COLOR_BGR2RGB)
                date_ = Image.fromarray(date_)

                sex_ = cv2.cvtColor(sex_, cv2.COLOR_BGR2RGB)
                sex_ = Image.fromarray(sex_)

                ID_ = cv2.cvtColor(ID_, cv2.COLOR_BGR2RGB)
                ID_ = Image.fromarray(ID_)

                origin_ = cv2.cvtColor(origin_, cv2.COLOR_BGR2RGB)
                origin_ = Image.fromarray(origin_)

                residence1_ = cv2.cvtColor(residence1_, cv2.COLOR_BGR2RGB)
                residence1_ = Image.fromarray(residence1_)

                residence2_ = cv2.cvtColor(residence2_, cv2.COLOR_BGR2RGB)
                residence2_ = Image.fromarray(residence2_)
                
                name = detector.predict(name_, return_prob=False)
                date = detector.predict(date_, return_prob=False)
                gender = detector.predict(sex_, return_prob=False)
                ID = detector.predict(ID_, return_prob=False)
                origin = detector.predict(origin_, return_prob=False)
                residence1 = detector.predict(residence1_, return_prob=False)
                residence2 = detector.predict(residence2_, return_prob=False)
                residence = residence1 + ', ' + residence2
                return render_template("viet_word.html", path1 = path1, name = name, ID = ID,\
                                date = date, gender = gender, origin = origin, residence = residence)

            elif mess == 'Chụp 1 hình thui':
                return jsonify({'message': 'Chup 1 hinh thoi'})
            
            elif mess == 'Không nhận diện được':
                return jsonify({'message': 'Khong nhan dien duoc'})

###
#### convert các file png, jpeg sang jpg
# for filename in os.listdir('train_dataset'):
#     if filename.endswith('.png') or filename.endswith('.jpeg'):
#         img = Image.open('train_dataset/' + filename)
#         rgb_im = img.convert('RGB')
#         rgb_im.save('train_dataset/' + filename[:-4] + '.jpg')
###


# Đây là kq mặt sau CCCD mới
@app.route("/submit_name", methods = ['GET', 'POST'])
def main3():
    if request.method == 'POST':
        img = request.files['my_image']
        path1 = img.filename 

        name = path1.split('.')[0]
        tail = path1.split('.')[1]
        if tail ==  'heic' or tail ==  'HEIC':
            # Usage example
            heic_file_path = path_a + path1
            opencv_image = heic_to_opencv(heic_file_path)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_a + name + '_.jpg', opencv_image)
            img_path = path_a + name + '_.jpg'
            path1 = name + '_.jpg'
            print('abc',path1)
            print(img_path)

        else:
            img_path = path_a + path1
            print(img_path) 
            img.save(img_path)


        
        tl, tr, br, bl, warped, orig, mess = detected2(img_path)
        if  mess == 'oke':
            # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
            img_path3 = path_c + path1 
           
            cv2.imwrite(img_path3, warped)

            #-------------------------------------
            img = cv2.imread(path_c + path1)
            # img = cv2.bilateralFilter(img,9,75,75)######

            datee = information2(img)
            h, w, _ = datee.shape
            datee = cv2.resize(datee, (w*2, h*2))

            img_path4 = path_d
           
            cv2.imwrite(img_path4 + 'a.jpg', datee)

            datee = cv2.cvtColor(datee, cv2.COLOR_BGR2RGB)
            datee = Image.fromarray(datee)
            
            datee_ = detector.predict(datee, return_prob=False)
            datee_ = convert_number(datee_)
            #-------------------------------------
            return render_template("viet_name.html", path1 = path1, datee_ = datee_)
        else:
            
            tl, tr, br, bl, warped, orig, mess = detected_star(img_path)
            if  mess == 'oke':
                # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
                img_path3 = path_c + path1 
            
                cv2.imwrite(img_path3, warped)

                #-------------------------------------
                img = cv2.imread(path_c + path1)
            
                datee = information2(img)
                h, w, _ = datee.shape
                datee = cv2.resize(datee, (w*2, h*2))

                img_path4 = path_d
            
                cv2.imwrite(img_path4 + 'a.jpg', datee)

                datee = cv2.cvtColor(datee, cv2.COLOR_BGR2RGB)
                datee = Image.fromarray(datee)
                
                datee_ = detector.predict(datee, return_prob=False)
                datee_ = convert_number(datee_)
                #-------------------------------------
                return render_template("viet_name.html", path1 = path1, datee_ = datee_)
            elif mess == 'Chụp 1 hình thui':
                return jsonify({'message': 'Chup 1 hinh thoi'})
            
            elif mess == 'Không nhận diện được':
                return jsonify({'message': 'Khong nhan dien duoc'})

###
###
###

# đây là kq quả phần đọc thông tin mặt trước cccd cũ
@app.route("/next_to_cccd_truoc", methods = ['GET', 'POST'])
def main4():
    if request.method == 'POST':
        img = request.files['my_image']
        path1 = img.filename 

        name = path1.split('.')[0]
        tail = path1.split('.')[1]
        if tail ==  'heic' or tail ==  'HEIC':
            # Usage example
            heic_file_path = path_a + path1
            opencv_image = heic_to_opencv(heic_file_path)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_a + name + '_.jpg', opencv_image)
            img_path = path_a + name + '_.jpg'
            path1 = name + '_.jpg'
            print('abc',path1)
            print(img_path)

        else:
            img_path = path_a + path1
            print(img_path) 
            img.save(img_path)

        # img_path = "C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/a/" + path1 
        # img_path = path_a + path1 

        # img.save(img_path)
        
        tl, tr, br, bl, warped, orig, mess = detected(img_path)
        if  mess == 'oke':
            
            # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
            img_path3 = path_c + path1 

            cv2.imwrite(img_path3, warped)

            #-------------------------------------
            img = cv2.imread(path_c + path1)
            name_, ID_, date_, sex_, origin1_, origin2_, residence1_, residence2_ = information3(img)
            img_path4 = path_d
            
            cv2.imwrite(img_path4 + 'a.jpg', name_)
            cv2.imwrite(img_path4 + 'b.jpg', ID_)
            cv2.imwrite(img_path4 + 'c.jpg', date_)
            cv2.imwrite(img_path4 + 'd.jpg', sex_)
            cv2.imwrite(img_path4 + 'e.jpg', origin1_)
            cv2.imwrite(img_path4 + 'e2.jpg', origin2_)
            cv2.imwrite(img_path4 + 'f.jpg', residence1_)
            cv2.imwrite(img_path4 + 'f2.jpg', residence2_)

            name_ = Image.fromarray(name_)

            date_ = cv2.cvtColor(date_, cv2.COLOR_BGR2RGB)
            date_ = Image.fromarray(date_)

            sex_ = cv2.cvtColor(sex_, cv2.COLOR_BGR2RGB)
            sex_ = Image.fromarray(sex_)

            ID_ = cv2.cvtColor(ID_, cv2.COLOR_BGR2RGB)
            ID_ = Image.fromarray(ID_)

            origin1_ = cv2.cvtColor(origin1_, cv2.COLOR_BGR2RGB)
            origin1_ = Image.fromarray(origin1_)

            origin2_ = cv2.cvtColor(origin2_, cv2.COLOR_BGR2RGB)
            origin2_ = Image.fromarray(origin2_)

            residence1_ = cv2.cvtColor(residence1_, cv2.COLOR_BGR2RGB)
            residence1_ = Image.fromarray(residence1_)

            residence2_ = cv2.cvtColor(residence2_, cv2.COLOR_BGR2RGB)
            residence2_ = Image.fromarray(residence2_)
            
            name = detector.predict(name_, return_prob=False)
            date = detector.predict(date_, return_prob=False)
            gender = detector.predict(sex_, return_prob=False)
            ID = detector.predict(ID_, return_prob=False)
            origin1 = detector.predict(origin1_, return_prob=False)
            if len(origin1) < 4 or "PRESENTING" in origin1:
                origin1 = ''
            if check_number(origin1): origin1 = ''

            origin2 = detector.predict(origin2_, return_prob=False)
            if len(origin2) < 4 or "PRESENTING" in origin2:
                origin2 = ''
            if check_number(origin2): origin2 = ''
            origin = origin1 + ', ' + origin2
            origin = origin.title()

            residence1 = detector.predict(residence1_, return_prob=False)
            residence2 = detector.predict(residence2_, return_prob=False)
            residence = residence1 + ', ' + residence2
            #-------------------------------------


            return render_template("cccd_truoc.html", path1 = path1, name = name, ID = ID,\
                                date = date, gender = gender, origin = origin, residence = residence)
        else:

            tl, tr, br, bl, warped, orig, mess = detected_star(img_path)
            if  mess == 'oke':
            
                # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
                img_path3 = path_c + path1 

                cv2.imwrite(img_path3, warped)

                #-------------------------------------
                img = cv2.imread(path_c + path1)
                name_, ID_, date_, sex_, origin1_, origin2_, residence1_, residence2_ = information3(img)
                img_path4 = path_d
                
                cv2.imwrite(img_path4 + 'a.jpg', name_)
                cv2.imwrite(img_path4 + 'b.jpg', ID_)
                cv2.imwrite(img_path4 + 'c.jpg', date_)
                cv2.imwrite(img_path4 + 'd.jpg', sex_)
                cv2.imwrite(img_path4 + 'e.jpg', origin1_)
                cv2.imwrite(img_path4 + 'e2.jpg', origin2_)
                cv2.imwrite(img_path4 + 'f.jpg', residence1_)
                cv2.imwrite(img_path4 + 'f2.jpg', residence2_)

                name_ = Image.fromarray(name_)

                date_ = cv2.cvtColor(date_, cv2.COLOR_BGR2RGB)
                date_ = Image.fromarray(date_)

                sex_ = cv2.cvtColor(sex_, cv2.COLOR_BGR2RGB)
                sex_ = Image.fromarray(sex_)

                ID_ = cv2.cvtColor(ID_, cv2.COLOR_BGR2RGB)
                ID_ = Image.fromarray(ID_)

                origin1_ = cv2.cvtColor(origin1_, cv2.COLOR_BGR2RGB)
                origin1_ = Image.fromarray(origin1_)

                origin2_ = cv2.cvtColor(origin2_, cv2.COLOR_BGR2RGB)
                origin2_ = Image.fromarray(origin2_)

                residence1_ = cv2.cvtColor(residence1_, cv2.COLOR_BGR2RGB)
                residence1_ = Image.fromarray(residence1_)

                residence2_ = cv2.cvtColor(residence2_, cv2.COLOR_BGR2RGB)
                residence2_ = Image.fromarray(residence2_)
                
                name = detector.predict(name_, return_prob=False)
                date = detector.predict(date_, return_prob=False)
                gender = detector.predict(sex_, return_prob=False)
                ID = detector.predict(ID_, return_prob=False)
                origin1 = detector.predict(origin1_, return_prob=False)
                if len(origin1) < 4 or "PRESENTING" in origin1:
                    origin1 = ''
                if check_number(origin1): origin1 = ''

                origin2 = detector.predict(origin2_, return_prob=False)
                if len(origin2) < 4 or "PRESENTING" in origin2:
                    origin2 = ''
                if check_number(origin2): origin2 = ''
                origin = origin1 + ', ' + origin2
                origin = origin.title()

                residence1 = detector.predict(residence1_, return_prob=False)
                residence2 = detector.predict(residence2_, return_prob=False)
                residence = residence1 + ', ' + residence2
                #-------------------------------------


                return render_template("cccd_truoc.html", path1 = path1, name = name, ID = ID,\
                                    date = date, gender = gender, origin = origin, residence = residence)

            elif mess == 'Chụp 1 hình thui':
                return jsonify({'message': 'Chup 1 hinh thoi'})
            
            elif mess == 'Không nhận diện được':
                return jsonify({'message': 'Khong nhan dien duoc'})

# Đây là kq mặt sau CCCD cũ
@app.route("/submit_cccd_sau", methods = ['GET', 'POST'])
def main7():
    if request.method == 'POST':
        img = request.files['my_image']
        path1 = img.filename 


        name = path1.split('.')[0]
        tail = path1.split('.')[1]
        if tail ==  'heic' or tail ==  'HEIC':
            # Usage example
            heic_file_path = path_a + path1
            opencv_image = heic_to_opencv(heic_file_path)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_a + name + '_.jpg', opencv_image)
            img_path = path_a + name + '_.jpg'
            path1 = name + '_.jpg'
            print('abc',path1)
            print(img_path)

        else:
            img_path = path_a + path1
            print(img_path) 
            img.save(img_path)
        # img_path = "C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/a/" + path1 
        # img_path = path_a + path1 

        # img.save(img_path)
        
        tl, tr, br, bl, warped, orig, mess = detected2(img_path)
        if  mess == 'oke':
            # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
            img_path3 = path_c + path1 
           
            cv2.imwrite(img_path3, warped)

            #-------------------------------------
            # img = cv2.imread('C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1)
            img = cv2.imread(path_c + path1)
        
            date1 = information7(img)

            img_path4 = path_d
            cv2.imwrite(img_path4 + 'a.jpg', date1)

            date1 = cv2.cvtColor(date1, cv2.COLOR_BGR2RGB)
            date1 = Image.fromarray(date1)

            
            date1 = detector.predict(date1, return_prob=False)
            
            #-------------------------------------


            return render_template("cccd_sau.html", path1 = path1, date1 = date1)

        elif mess == 'Chụp 1 hình thui':
            return jsonify({'message': 'Chup 1 hinh thoi'})
        
        elif mess == 'Không nhận diện được':
            return jsonify({'message': 'Khong nhan dien duoc'})

###
###
###

#----------------------------------------------------------------
#----------------------------------------------------------------
###
###
###

# đây là kq quả phần đọc thông tin mặt trước cmnd cũ
@app.route("/next_to_cmnd_truoc", methods = ['GET', 'POST'])
def main5():
    if request.method == 'POST':
        img = request.files['my_image']
        path1 = img.filename 

        name = path1.split('.')[0]
        tail = path1.split('.')[1]
        if tail ==  'heic' or tail ==  'HEIC':
            # Usage example
            heic_file_path = path_a + path1
            opencv_image = heic_to_opencv(heic_file_path)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_a + name + '_.jpg', opencv_image)
            img_path = path_a + name + '_.jpg'
            path1 = name + '_.jpg'
            print('abc',path1)
            print(img_path)

        else:
            img_path = path_a + path1
            print(img_path) 
            img.save(img_path)
        # img_path = "C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/a/" + path1 
        # img_path = path_a + path1 

        # img.save(img_path)
        
        tl, tr, br, bl, warped, orig, mess = detected2(img_path)
        if  mess == 'oke':
            
            # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
            img_path3 = path_c + path1 

            cv2.imwrite(img_path3, warped)

            #-------------------------------------
            img = cv2.imread(path_c + path1)
            name1_, name2_, ID_, date_, origin1_, origin2_, residence1_, residence2_ = information4(img)
            # img_path4 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/d/'
            img_path4 = path_d

            cv2.imwrite(img_path4 + 'a.jpg', name1_)
            cv2.imwrite(img_path4 + 'a2.jpg', name2_)
            cv2.imwrite(img_path4 + 'b.jpg', ID_)
            cv2.imwrite(img_path4 + 'c.jpg', date_)
            # cv2.imwrite(img_path4 + 'd.jpg', sex_)
            cv2.imwrite(img_path4 + 'e.jpg', origin1_)
            cv2.imwrite(img_path4 + 'e2.jpg', origin2_)
            cv2.imwrite(img_path4 + 'f.jpg', residence1_)
            cv2.imwrite(img_path4 + 'f2.jpg', residence2_)

            name1_ = cv2.cvtColor(name1_, cv2.COLOR_BGR2RGB)
            h, w, _ = name1_.shape
            name1_ = cv2.resize(name1_ , (w*5, h*5))
            name1_ = Image.fromarray(name1_)
            


            name2_ = cv2.cvtColor(name2_, cv2.COLOR_BGR2RGB)
            h, w, _ = name2_.shape
            name2_ = cv2.resize(name2_ , (w*5, h*5))
            name2_ = Image.fromarray(name2_)

            date_ = cv2.cvtColor(date_, cv2.COLOR_BGR2RGB)
            date_ = Image.fromarray(date_)

            ID_ = cv2.cvtColor(ID_, cv2.COLOR_BGR2RGB)
            ID_ = Image.fromarray(ID_)

            origin1_ = cv2.cvtColor(origin1_, cv2.COLOR_BGR2RGB)
            origin1_ = Image.fromarray(origin1_)

            origin2_ = cv2.cvtColor(origin2_, cv2.COLOR_BGR2RGB)
            h, w, _ = origin2_.shape
            origin2_ = cv2.resize(origin2_ , (w*5, h*5))
            origin2_ = Image.fromarray(origin2_)

            residence1_ = cv2.cvtColor(residence1_, cv2.COLOR_BGR2RGB)
            residence1_ = Image.fromarray(residence1_)

            residence2_ = cv2.cvtColor(residence2_, cv2.COLOR_BGR2RGB)
            residence2_ = Image.fromarray(residence2_)
            
            name1 = detector.predict(name1_, return_prob=False)
            if len(name1) < 7 or "PRESENTING" in name1:
                name1 = ''
            if check_number(name1): name1 = ''

            name2 = detector.predict(name2_, return_prob=False)
            if len(name2) < 7 or 'PRESENTING' in name2:
                name2 = ''
            if check_number(name2): name2 = ''

            name = name1 + ' ' + name2

            date = detector.predict(date_, return_prob=False)
            date = convert_number(date)

            ID = detector.predict(ID_, return_prob=False)
            origin1 = detector.predict(origin1_, return_prob=False)
            origin2 = detector.predict(origin2_, return_prob=False)
            origin = origin1 + ', ' + origin2
            residence1 = detector.predict(residence1_, return_prob=False)
            residence2 = detector.predict(residence2_, return_prob=False)
            residence = residence1 + ', ' + residence2
            #-------------------------------------


            return render_template("cmnd_truoc.html", path1 = path1, name = name, ID = ID,\
                                date = date, origin = origin, residence = residence)

        elif mess == 'Chụp 1 hình thui':
            return jsonify({'message': 'Chup 1 hinh thoi'})
        
        elif mess == 'Không nhận diện được':
            return jsonify({'message': 'Khong nhan dien duoc'})


####
####

# kết quả mặt sau CMND
@app.route("/submit_cmnd_sau", methods = ['GET', 'POST'])
def main6():
    if request.method == 'POST':
        img = request.files['my_image']
        path1 = img.filename 

        name = path1.split('.')[0]
        tail = path1.split('.')[1]
        if tail ==  'heic' or tail ==  'HEIC':
            # Usage example
            heic_file_path = path_a + path1
            opencv_image = heic_to_opencv(heic_file_path)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_a + name + '_.jpg', opencv_image)
            img_path = path_a + name + '_.jpg'
            path1 = name + '_.jpg'
            print('abc',path1)
            print(img_path)

        else:
            img_path = path_a + path1
            print(img_path) 
            img.save(img_path)

        # img_path = "C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/a/" + path1 
        # img_path = path_a + path1 

        # img.save(img_path)
        
        tl, tr, br, bl, warped, orig, mess = detected2(img_path)
        if  mess == 'oke':
            # img_path3 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/c/' + path1 
            img_path3 = path_c + path1 
           
            cv2.imwrite(img_path3, warped)

            #-------------------------------------
            img = cv2.imread(path_c + path1)
            date1 = information6(img)
            

            # img_path4 = 'C:/Users/ADMIN/Desktop/Thay_Lam/CCCD/web_app/static/d/'
            img_path4 = path_d
           
            cv2.imwrite(img_path4 + 'a.jpg', date1)

            date1 = cv2.cvtColor(date1, cv2.COLOR_BGR2RGB)
            date1 = Image.fromarray(date1)
            
            date1 = detector.predict(date1, return_prob=False)
            #-------------------------------------


            return render_template("cmnd_sau.html", path1 = path1, date1 = date1)

        elif mess == 'Chụp 1 hình thui':
            return jsonify({'message': 'Chup 1 hinh thoi'})
        
        elif mess == 'Không nhận diện được':
            return jsonify({'message': 'Khong nhan dien duoc'})

###
###
###


@app.route("/next_to_viet_word", methods = ['POST'])
def viet_word_page():
    if request.form.get('btn1') == 'Read1':
        return render_template("viet_word.html")

@app.route("/next_to_viet_name", methods = ['POST'])
def viet_name_page():
    if request.form.get('btn2') == 'Read2':
        return render_template("viet_name.html")
#----------------------------------------
@app.route("/next_to_cccd_truoc1", methods = ['POST'])
def cccd_truoc_next():
    if request.form.get('btn3') == 'Read3':
        return render_template("cccd_truoc.html")

@app.route("/next_to_cccd_sau", methods = ['POST'])
def cccd_sau_next():
    if request.form.get('btn4') == 'Read4':
        return render_template("cccd_sau.html")
#----------------------------------------------
@app.route("/next_to_cmnd_truoc1", methods = ['POST'])
def cmnd_truoc_next():
    if request.form.get('btn5') == 'Read5':
        return render_template("cmnd_truoc.html")

@app.route("/next_to_cmnd_sau", methods = ['POST'])
def cmnd_sau_next():
    if request.form.get('btn6') == 'Read6':
        return render_template("cmnd_sau.html")












if __name__ == '__main__':
    app.run(debug = False, port=6006, host='0.0.0.0')
    # app.run(debug = True)
    

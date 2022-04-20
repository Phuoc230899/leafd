from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from flask_sqlalchemy import SQLAlchemy
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from datetime import datetime
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,flash
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
import pickle
# Define a flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydb.db'
db = SQLAlchemy(app)
user_name = ['']

class Accounts(db.Model):
    Id_Account = db.Column(db.Integer, primary_key= True)
    User_Name = db.Column(db.String(200), nullable=False )
    User_Account = db.Column(db.String(200), unique=True)
    Password = db.Column(db.String(200), nullable=False)
    Date_Created = db.Column(db.DateTime, default=datetime.utcnow)

class Topic(db.Model):
    Id_Topic = db.Column(db.Integer, primary_key= True)
    Id_Account = db.Column(db.Integer, db.ForeignKey('accounts.Id_Account'))
    Title = db.Column(db.String(200))
    Description = db.Column(db.String(2000), nullable=False)
    Date_Created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<topics %r>' % self.ID_Topic

class Comments(db.Model):
    Id_Topic = db.Column(db.Integer, db.ForeignKey('topic.Id_Topic'))
    Id_Account = db.Column(db.Integer, db.ForeignKey('accounts.Id_Account'))
    Id_Comment = db.Column(db.Integer, primary_key= True)
    Description = db.Column(db.String(2000), nullable=False)
    Date_Created = db.Column(db.DateTime, default=datetime.utcnow)

class Products(db.Model):
    Id_Product = db.Column(db.Integer, primary_key= True)
    Price = db.Column(db.Integer)
    Product_Name = db.Column(db.String(2000))
    Quantity = db.Column(db.Integer)
    Date = db.Column(db.DateTime, default=datetime.utcnow)
    Link_Image = db.Column(db.String(2000))

class Cart(db.Model):
    Id = db.Column(db.Integer, primary_key= True)
    Id_Account = db.Column(db.Integer, db.ForeignKey('accounts.Id_Account'))
    Product_Name = db.Column(db.String(2000), db.ForeignKey('products.Product_Name'))
    Quantity = db.Column(db.Integer)
    Link_Image = db.Column(db.String(2000))
    Price = db.Column(db.Integer)

# Model saved with Keras model.save()
MODEL_PATH ='best.hdf5'
classes = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight'
        ,'Potato___healthy','Potato___Late_blight','Tomato__Target_Spot','Tomato__Tomato_mosaic_virus'
        ,'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_healthy'
        ,'Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite' ]
# Load your trained model
model = load_model(MODEL_PATH)
# model = pickle.load(open('./cnn_model.pkl', 'rb'))
# default_image_size = tuple((256, 256))
# def convert_image_to_array(image_dir):
#     try:
#         image = cv2.imread(image_dir)
#         if image is not None :
#             image = cv2.resize(image, default_image_size)   
#             return img_to_array(image)
#         else :
#             return np.array([])
#     except Exception as e:
#         print(f"Error : {e}")
#         return None

def preprocess_image(img):
        if (img.shape[0] != 224 or img.shape[1] != 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
        img = (img/127.5)
        img = img - 1
        img = np.expand_dims(img, axis=0)
        return img



def model_predict(img_path, model):
#     print(img_path)
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     ## Scaling
#     x=x/255
#     x = np.expand_dims(x, axis=0)
   

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#    # x = preprocess_input(x)

#     preds = model.predict(x)
#     preds=np.argmax(preds, axis=1)
#     if preds==0:
#         preds="The leaf is diseased cotton leaf"
#     elif preds==1:
#         preds="The leaf is diseased cotton plant"
#     elif preds==2:
#         preds="The leaf is fresh cotton leaf"
#     else:
#         preds="The leaf is fresh cotton plant"

    img = cv2.imread(img_path)
    pred = model.predict(preprocess_image(img))
    result = classes[np.argmax(pred)]        

    # im=convert_image_to_array(img_path)
    # np_image_li = np.array(im, dtype=np.float16) / 225.0
    # npp_image = np.expand_dims(np_image_li, axis=0)
    # result=model.predict(npp_image)
    return str(result)


@app.route('/single-product')
def singleprod():
    un = user_name[-1]
    return render_template('single-product.html',username=un)

@app.route('/detect', methods=['GET'])
def predict():
    un = user_name[-1]
    return render_template('index.html',username = un)

@app.route('/')
def index():
    # Main page
    un = user_name[-1]
    return render_template('homepage.html',username=un)

@app.route('/homepage')
def homepage():
    # Main page
    un = user_name[-1]
    return render_template('homepage.html',username = un)

@app.route('/dinhduong')
def dinhduong():
    un = user_name[-1]
    return render_template('dinhduong.html',username=un)

@app.route('/checkout')
def checkout():
    un = user_name[-1]
    return render_template('checkout.html',username=un)

@app.route('/Admin')
def adminmanager():
    return render_template('adminDashBoard.html')

@app.route('/manageAccounts')
def manageAccounts():
    accounts = Accounts.query.order_by(Accounts.Date_Created).all()
    return render_template('manageAccount.html',accounts = accounts)

@app.route('/managetopic')
def manageTopic():
    topics = Topic.query.order_by(Topic.Date_Created).all()
    return render_template('manageTopic.html',topics = topics)

@app.route('/deleteAccount/<int:Id_Account>')
def delete_account(Id_Account):
    Account_to_delete = Accounts.query.get_or_404(Id_Account)

    try:
        db.session.delete(Account_to_delete)
        db.session.commit()
        return redirect('/manageAccounts')
    except:
        return 'There was a problem deleting that task'

@app.route('/deleteTopic/<int:Id_Topic>')
def delete_topic(Id_Topic):
    topic_to_delete = Topic.query.get_or_404(Id_Topic)

    try:
        db.session.delete(topic_to_delete)
        db.session.commit()
        return redirect('/managetopic')
    except:
        return 'There was a problem deleting that task'

@app.route('/managetopic')
def managetopic():
    return render_template('manageTopic.html')

@app.route('/product')
def product():
    products = Products.query.order_by(Products.Id_Product).all()
    un = user_name[-1]
    return render_template('product.html',products = products,username = un)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    
    if request.method == 'POST':
        if request.form['useraccount'] == "admin@gmail.com" and request.form['password'] == "admin":
            return redirect("./Admin")
        useraccountcheck = Accounts.query.filter_by(User_Account = str(request.form['useraccount'])).first()
        if useraccountcheck is not None: 
            password =  useraccountcheck.Password
            if str(request.form['password']) != str(password):
                error = 'Sai tên đăng nhập hoặc mật khẩu !!!.'
            else:
                user_name.append(str(useraccountcheck.User_Name))
                return redirect("./homepage")
        else:
            error = 'Sai tên đăng nhập hoặc mật khẩu !!!.'
    return render_template('login.html')

@app.route('/signup', methods = ['POST', 'GET'])
def register():
    error = None
    now = datetime.now().date()
    if request.method == 'POST':
        useraccountcheck = Accounts.query.filter_by(User_Account = str(request.form['useraccount'])).first()
        username = request.form['username']
        password = request.form['password']
        useraccount = request.form['useraccount']
        repassword = request.form['repassword']
        if useraccountcheck is None:
            if repassword == password : 
                new_user = Accounts(User_Name = username,Password = password,User_Account = useraccount,Date_Created = now)
                db.session.add(new_user)
                db.session.commit()
                return redirect('/login')
            else:
                error = 'Mật khẩu không trùng khớp!!!'
        else:
            error = "User account exist!"
    return render_template('signup.html',error = error)


@app.route('/product/<int:Id_Product>',methods = ['POST', 'GET'])
def singleproduct(Id_Product):
    message = None   
    if request.method == 'POST':
        product = Products.query.filter_by(Id_Product = Id_Product).first()
        link_image = product.Link_Image
        product_name = product.Product_Name
        quantity = request.form['quantity']
        price = product.Price
        check = Cart.query.filter_by(Product_Name = product_name).first()
        if quantity == '':
            quantity = 1
        if check == None:
            prod_add_to_cart = Cart(Id_Account = 1,Product_Name = product_name,Quantity = quantity,Link_Image = link_image,Price = price)
            db.session.add(prod_add_to_cart)
            db.session.commit()
            message = "Success!"
        else:
            check.Quantity = check.Quantity + int(quantity)
            db.session.commit()
        return render_template('single-product.html',product = product,message = message) 
    else :
        product = Products.query.filter_by(Id_Product = Id_Product).first()
        return render_template('single-product.html',product = product) 


@app.route('/cart')
def cart():
    un = user_name[-1]
    sum = 0
    items = Cart.query.filter_by(Id_Account = 1).all()
    for item in items:
        sum = sum+int(item.Price)*int(item.Quantity)
    return render_template('cart.html',items = items,sum = sum,username = un)

@app.route('/topic', methods = ['POST','GET'])
def uptopic():
    error = None
    now = datetime.now()
    un = user_name[-1]
    if request.method == 'POST':
        title = request.form['title']
        decription = request.form['description']
        new_topic = Topic(Id_Account = 1,Title = title,Description=decription, Date_Created = now)
        db.session.add(new_topic)
        db.session.commit()
        return redirect('/topic')
    else:
        topics = Topic.query.order_by(Topic.Date_Created).all()
        return render_template('post.html', topics=topics,username=un)
        #error = "Lỗi ko thêm topic được!!!"

    return render_template('post.html',username=un)


@app.route('/topic/<int:Id_Topic>', methods = ['POST','GET'])
def comment(Id_Topic):
    error = None
    now = datetime.now()
    if request.method == 'POST':
        comment_content = request.form['comment']
        new_comment = Comments(Id_Account = 1,Description=comment_content,Id_Topic= Id_Topic,Date_Created = now)
        db.session.add(new_comment)
        db.session.commit()
        return redirect('/topic/'+str(Id_Topic))
    else:
        topic = Topic.query.filter_by(Id_Topic = Id_Topic).all()
        comments = Comments.query.filter_by(Id_Topic = Id_Topic).all()
        return render_template('post_detail.html', comments=comments,topic = topic)
        #error = "Lỗi ko thêm comment được!!!"
    return render_template('post_detail.html')

if __name__ == '__main__':
    app.run(host="localhost",port=5001,debug=True)

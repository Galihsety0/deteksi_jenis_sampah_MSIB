from flask import Flask, render_template, request, send_file, jsonify, redirect, Response
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
# Flatten, Dense, Activation, Dropout,LeakyReLU
# from PIL import Image
# from fungsi import make_model
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
# import keras.applications.mobilenet_v2 as mobilenetv2
# import tensorflow.keras as keras
# from tensorflow.keras.applications import mobilenet_v2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


# ====== FLASK SETUP ======

UPLOAD_FOLDER = 'C:\\Users\\galih\\Downloads\\deteksi_jenis_sampah_MSIB\\test images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# app   = Flask(__name__, static_url_path='/static')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SECRET_KEY'] = 'ini secret key KAMI'

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

model = None

NUM_CLASSES = 12
cifar10_classes = ["paper", "cardboard", "plastic", "metal", "trash", "battery", "shoes", "clothes", "green-glass", "brown-glass", "white-glass", "biological"]

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def index():
    return render_template('index.html')

# [Routing untuk Halaman About]
@app.route("/about")
def about():
    return render_template('about.html')

# [Routing untuk Halaman team]
@app.route("/team")
def team():
    return render_template('team.html')

# [Routing untuk Halaman apikasi]
@app.route("/aplikasi")
def aplikasi():
    return render_template('aplikasi.html')


@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi
    hasil_prediksi = '(none)'
    gambar_prediksi = '(none)'

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)

    # Periksa apakah ada file yg dipilih untuk diupload
    if filename != '':

        # Set/mendapatkan extension dan path dari file yg diupload
        file_ext = os.path.splitext(filename)[1]
        gambar_prediksi = '/static/images/uploads/' + filename

        # Periksa apakah extension file yg diupload sesuai (jpg)
        if file_ext in app.config['UPLOAD_EXTENSIONS']:

            # Simpan Gambar
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_PATH'], filename))

            # Memuat Gambar
            lok = '.' + gambar_prediksi
            gmbr = ts.keras.utils.load_img(lok, target_size=(150, 150))
            x = ts.keras.utils.img_to_array(gmbr)
            x = np.expand_dims(x, axis=0)
            gmbr = np.vstack([x])

            # Prediksi Gambar
            kelas, df = PredGambar(gmbr)
            hasil_prediksi = kelas

            # Return hasil prediksi dengan format JSON
            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })
        else:
            # Return hasil prediksi dengan format JSON
            gambar_prediksi = '(none)'
            return jsonify({
                "prediksi": hasil_prediksi,
                "gambar_prediksi": gambar_prediksi
            })
        

if __name__ == "__main__":
    # Load model yang telah ditraining
    # model = make_model()
    # model.load_weights("fix_garbage_classification_model.h5")
    app.run(host="localhost", port=5000, debug=True)


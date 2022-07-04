from flask import Flask,request, render_template

import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pickle
import numpy as np
from scipy.io.wavfile import read
from matplotlib import pyplot as plt

app = Flask(__name__)

@app.route("/" ,methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def upload():
        
    if request.method == 'POST':
        audiofile=request.files['file']
        audio_pa=audiofile.filename
        audiofile.save(audio_pa)

        svcmodel = pickle.load(open(r"C:\Users\ADMIN\Downloads\accentsvm.sav", 'rb'))
        vggmodel = load_model(r'C:\Users\ADMIN\Downloads\accentvgg16.h5')
        rd_data = read(audio_pa)
        ad_features = rd_data[1]
        plt.plot(ad_features)
        plt.savefig(r"C:\Users\ADMIN\Downloads\TamilAudio\val1.png")
        plt.close()
        img1 =  keras.utils.load_img(r"C:\Users\ADMIN\Downloads\TamilAudio\val1.png", target_size=(224, 224))
        x1 = image.img_to_array(img1)
        x1 = np.expand_dims(x1, axis=0)
        x1 = preprocess_input(x1)
        flatten1 = vggmodel.predict(x1)
        pred = svcmodel.predict([flatten1[0]])

        if pred==0:
            result = "Telugu"
        elif pred==1:
            result = "Bangla"
        elif pred==2:
            result = "Odiya"
        elif pred==3:
            result = "Malayalam"
        else:
            result = "Tamil"
        return result
    return None
    
    
if __name__ == "__main__":
    app.run(debug=True)
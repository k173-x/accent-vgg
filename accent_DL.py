# -*- coding: utf-8 -*-
"""Accent Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T6sCHDFnE5-7YS01ouosINxn_aKHoxwg
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Import section
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import math, random

# Create a seed for reproducibility
seed = 1337
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed) 
random.seed(1337)
tf.random.set_seed(seed)

!pip install gdown

print("Downloading dataset... standby!")
!gdown https://drive.google.com/uc?id=1a5cN4GwzsngrpYP230hzM58I8BhtB8et #-> Core
#!gdown https://drive.google.com/uc?id=1NO1NKQSpyq3DMLEwiqA-BHIqXli8vtIL #-> Full extended

print("Download complete! Extraction in progress...")
!tar -xvzf ./accentdb_core.tar.gz #Change name here...
print("Deleting tar!")
!rm -rf ./accentdb_core.tar.gz

!print("Downloading custom dataset...standby!")
!gdown https://drive.google.com/uc?id=1DK5MvtJu5vx_CW8dyfLAhDCfhX9wnKJ- #-> Custom

!print("Extraction in progress")
!unzip ./"TamilAudio (2).zip" -d ./data/Tamil

!pip install librosa

import librosa
import librosa.display
audio_data = './data/telugu/speaker_01/telugu_s01_745.wav'
x , sr = librosa.load(audio_data,sr=44100)
print(type(x), type(sr))
print(x.shape, sr)

import IPython.display as ipd

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

audio_fpath = "./data"
audio_clips = getListOfFiles(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display as ld
plt.figure(figsize=(14, 5))
ld.waveshow(x, sr=sr)

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')

spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
plt.figure(figsize=(12, 4))
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
plt.figure(figsize=(15, 9))
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))

x, sr = librosa.load(audio_data)
#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

from scipy.io.wavfile import read
from os import walk
import time
print("Converting dataset.... This will take at least 10 minutes! Please wait!!")
if not os.path.exists("teluguPlot"):
    os.makedirs("teluguPlot")
if not os.path.exists("banglaPlot"):
    os.makedirs("banglaPlot")
if not os.path.exists("malayalamPlot"):
    os.makedirs("malayalamPlot")
if not os.path.exists("odiyaPlot"):
    os.makedirs("odiyaPlot")
if not os.path.exists("tamilPlot"):
    os.makedirs("tamilPlot")
files_size = 100

lang_wavs = []
Telugu_lang_wav = []
i = 1
print("Converting set A-1")
for (_,_,filenames) in walk('data/telugu/speaker_01'):
    lang_wavs.extend(filenames)
    break
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Telugu_lang_wav.append("./data/telugu/speaker_01/" + wavs)
      input_data = read("./data/telugu/speaker_01/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("teluguPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

time.sleep(1)

print("Converting set A-2")
lang_wavs = []
for (_,_,filenames) in walk('data/telugu/speaker_02'):
    lang_wavs.extend(filenames)
    break
i = 1
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Telugu_lang_wav.append("./data/telugu/speaker_02/" + wavs)
      input_data = read("./data/telugu/speaker_02/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("teluguPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

time.sleep(1)
print("Converting set B-1")
i = 1
Bangla_lang_wav = []
lang_wavs = []
for (_,_,filenames) in walk('data/bangla/speaker_01'):
    lang_wavs.extend(filenames)
    break
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Bangla_lang_wav.append("./data/bangla/speaker_01/" + wavs)
      input_data = read("./data/bangla/speaker_01/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("banglaPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

time.sleep(1)
print("Converting set B-2")
lang_wavs = []
i = 1
for (_,_,filenames) in walk('data/bangla/speaker_02'):
    lang_wavs.extend(filenames)
    break
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Bangla_lang_wav.append("./data/bangla/speaker_02/" + wavs)
      input_data = read("./data/bangla/speaker_02/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("banglaPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

time.sleep(1)
print("Converting set C-1")
Malayalam_lang_wav = []
lang_wavs = []
i = 1
for (_,_,filenames) in walk('data/malayalam/speaker_01'):
    lang_wavs.extend(filenames)
    break
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Malayalam_lang_wav.append("./data/malayalam/speaker_01/" + wavs)
      input_data = read("./data/malayalam/speaker_01/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("malayalamPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

time.sleep(1)
print("Converting set C-2")
lang_wavs = []
i = 1
for (_,_,filenames) in walk('data/malayalam/speaker_02'):
    lang_wavs.extend(filenames)
    break
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Malayalam_lang_wav.append("./data/malayalam/speaker_02/" + wavs)
      input_data = read("./data/malayalam/speaker_02/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("malayalamPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

time.sleep(1)
lang_wavs = []
Odiya_lang_wav = []
print("Converting set D-1")
i = 1
for (_,_,filenames) in walk('data/odiya/speaker_01'):
    lang_wavs.extend(filenames)
    break
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Odiya_lang_wav.append("./data/odiya/speaker_01/" + wavs)
      input_data = read("./data/odiya/speaker_01/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("odiyaPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

time.sleep(1)
lang_wavs = []
print("Converting set D-2")
i = 1
for (_,_,filenames) in walk('data/odiya/speaker_02'):
    lang_wavs.extend(filenames)
    break
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Odiya_lang_wav.append("./data/odiya/speaker_02/" + wavs)
      input_data = read("./data/odiya/speaker_02/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("odiyaPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

time.sleep(1)
lang_wavs = []
Tamil_lang_wav = []
print("Converting set E-1")
i = 1
for (_,_,filenames) in walk('data/Tamil/TamilAudio'):
    lang_wavs.extend(filenames)
    break
for wavs in lang_wavs:
    if(i!=files_size):
      # read audio samples
      Tamil_lang_wav.append("./data/Tamil/TamilAudio/" + wavs)
      input_data = read("./data/Tamil/TamilAudio/" + wavs)
      audio = input_data[1]
      # plot the first 1024 samples
      plt.plot(audio)
      # label the axes
      plt.ylabel("Amplitude")
      plt.xlabel("Time")
      # set the title
      # plt.title("Sample Wav")
      # display the plot
      plt.savefig("tamilPlot/" + wavs.split('.')[0] + '.png')
      # plt.show()
      plt.close('all')
      i = i + 1

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

def get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    return list(flatten[0])

X = []
y = []

lang1_plots = []
for (_,_,filenames) in os.walk('teluguPlot'):
    lang1_plots.extend(filenames)
    break

for aplot in lang1_plots:
    X.append(get_features('teluguPlot/' + aplot))
    y.append(0)

lang2_plots = []
for (_,_,filenames) in os.walk('banglaPlot'):
    lang2_plots.extend(filenames)
    break

for bplot in lang2_plots:
    X.append(get_features('banglaPlot/' + bplot))
    y.append(1)

lang3_plots = []
for (_,_,filenames) in os.walk('odiyaPlot'):
    lang3_plots.extend(filenames)
    break

for cplot in lang3_plots:
    X.append(get_features('odiyaPlot/' + cplot))
    y.append(2)

lang4_plots = []
for (_,_,filenames) in os.walk('malayalamPlot'):
    lang4_plots.extend(filenames)
    break

for dplot in lang4_plots:
    X.append(get_features('malayalamPlot/' + dplot))
    y.append(3)

lang5_plots = []
for (_,_,filenames) in os.walk('tamilPlot'):
    lang5_plots.extend(filenames)
    break

for dplot in lang5_plots:
    X.append(get_features('tamilPlot/' + dplot))
    y.append(4)

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

# get the accuracy
print (accuracy_score(y_test, predicted))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test,predicted)
print(cm)
sns.heatmap(cm,annot=True)

print(classification_report(y_test,predicted))

#Validation Phase

rd_data = read("/content/data/Tamil/TamilAudio/101.wav")
ad_features = rd_data[1]
plt.plot(ad_features)
plt.savefig("/content/val1.png")
plt.close()
img1 = image.load_img("/content/val1.png", target_size=(224, 224))
x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)
flatten1 = model.predict(x1)
pred = clf.predict([flatten1[0]])

if pred==0:
  print("Language : Telugu")
elif pred==1:
  print("Language : Bangla")
elif pred==2:
  print("Language : Odiya")
elif pred==3:
  print("Language : Malayalam")
else:
  print("Language : Tamil")

# To save the particular model in .h5 format
import tensorflow as tf
from tensorflow.keras.models import load_model
model.save('accentvgg16.h5')

# save the model to disk
import pickle
filename = 'accentsvm.sav'
pickle.dump(clf, open(filename, 'wb'))

"""# MFCC Feature Extraction"""

from librosa.feature import mfcc
mfcc_t = mfcc(x, sr)
plt.plot(mfcc_t)

import librosa
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten').output)

def get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    return list(flatten[0])

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features.tolist()

X1 = []
X2 = []
y = []
  
lang1_plots = []
for (_,_,filenames) in os.walk('teluguPlot'):
    lang1_plots.extend(filenames)
    break

for awav in Telugu_lang_wav:
    X1.append(features_extractor(awav))
    y.append(0)

for aplot in lang1_plots:
    l1=get_features('teluguPlot/' + aplot)
    X2.append(l1)

for bwav in Bangla_lang_wav:
    X1.append(features_extractor(bwav))
    y.append(1)

lang2_plots = []
for (_,_,filenames) in os.walk('banglaPlot'):
    lang2_plots.extend(filenames)
    break

for bplot in lang2_plots:
    l2=get_features('banglaPlot/' + bplot)
    X2.append(l2)

for cwav in Odiya_lang_wav:
    X1.append(features_extractor(cwav))
    y.append(2)

lang3_plots = []
for (_,_,filenames) in os.walk('odiyaPlot'):
    lang3_plots.extend(filenames)
    break

for cplot in lang3_plots:
    l3=get_features('odiyaPlot/' + cplot)
    X2.append(l3)

for dwav in Malayalam_lang_wav:
    X1.append(features_extractor(dwav))
    y.append(3)

lang4_plots = []
for (_,_,filenames) in os.walk('malayalamPlot'):
    lang4_plots.extend(filenames)
    break

for dplot in lang4_plots:
    l4=get_features('malayalamPlot/' + dplot)
    X2.append(l4)

for ewav in Tamil_lang_wav:
    X1.append(features_extractor(ewav))
    y.append(4)

lang5_plots = []
for (_,_,filenames) in os.walk('tamilPlot'):
    lang5_plots.extend(filenames)
    break

for eplot in lang5_plots:
    l5=get_features('tamilPlot/' + eplot)
    X2.append(l5)

dataCoeff = pd.DataFrame(X1)
dataCoeff.head()

dataCoeff2= pd.DataFrame(X2)
dataCoeff2.head()

from sklearn.decomposition import PCA
pca = PCA(n_components=40, svd_solver='full')
newfeat = pca.fit_transform(dataCoeff2)

pca1 = PCA(n_components=35, svd_solver='arpack')
newfeat2 = pca1.fit_transform(dataCoeff2)

dataCoeff3 = pd.DataFrame(newfeat2)

dataCoeff2 = dataCoeff2.iloc[0:0]
dataCoeff2 = pd.DataFrame(newfeat)
dataCoeff2.head()

dataCoeff = dataCoeff.add_prefix('m_')
dataCoeff2 = dataCoeff2.add_prefix('v_')

dataCoeff

dataCoeff2

X_new = pd.concat([dataCoeff,dataCoeff2], axis=1, join='inner')
X_new

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=42, stratify=y)

np.shape(X)

np.ndim(X)

clf = LinearSVC(random_state=42, tol=1e-5)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

# get the accuracy
print (accuracy_score(y_test, predicted))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test,predicted)
print(cm)
sns.heatmap(cm,annot=True)

print(classification_report(y_test,predicted))

"""# Reduced VGG16

"""

X_red = dataCoeff3
X_train, X_test, y_train, y_test = train_test_split(X_red, y, test_size=0.25, random_state=42, stratify=y)

clf = LinearSVC(random_state=42, tol=1e-5)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

# get the accuracy
print (accuracy_score(y_test, predicted))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test,predicted)
print(cm)
sns.heatmap(cm,annot=True)

print(classification_report(y_test,predicted))


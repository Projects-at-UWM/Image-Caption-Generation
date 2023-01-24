import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.utils import np_utils
import random
from tensorflow.keras.preprocessing import sequence, image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import decode_predictions
from gensim.models import Word2Vec,KeyedVectors
import re
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.image import img_to_array


def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png"),("jpeg files","*.jpg")))
    basewidth = 250 # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def classify():
    global category
    #original = Image.open(image_data)
    original = image.load_img(image_data, target_size=(224,224,3))
    #original = original.resize((224, 224,3
    original = img_to_array(original)
    original = np.expand_dims(original, axis=0)
    resnet = ResNet50(include_top=False,weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',input_shape=(224,224,3),pooling='avg')
    test_img = resnet.predict(original).reshape(2048)
    with open('OUTPUT/word_2_indices.pickle', 'rb') as handle:
        word_2_indices_data = pickle.load(handle)
    word_2_indices = word_2_indices_data
    with open('OUTPUT/indices_2_word.pickle', 'rb') as handle:
        indices_2_word_data = pickle.load(handle)
    indices_2_word = indices_2_word_data
    max_len = 40
    start_word = ["<start>"]
    while True:
        par_caps = [word_2_indices[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model1.predict([np.array([test_img]), np.array(par_caps)])
        word_pred = indices_2_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
    label= ' '.join(start_word[1:-1])
    model_w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True,limit=1000000)
    description = label
    description = description.replace(r'\d+','')
    spec_chars = ["!",'"',"#","%","&","'","(",")",
                "*","+",",","-",".","/",":",";","<",
                "=",">","?","@","[","\\","]","^","_",
                "`","{","|","}","~","–"]
    for char in spec_chars:
        description = description.replace(char, ' ')
    word_list_t = description.lower().split() 
    filtered_words = [word for word in word_list_t if word not in stopwords.words('english')]
    text = ' '.join(filtered_words)
    description = text
    description_tokens = list(description.split(" "))
    token_list=[]
    for i in description_tokens:
        if len(i) > 3 and i in model_w2v.vocab:
            token_list.append(i)
        else:
            continue
    description_tokens_filtered = token_list
    _arrays = np.zeros((1, 300))
    vec = np.zeros(300).reshape((1, 300))
    count = 0
    for word in description_tokens_filtered:
        vec += model_w2v[word].reshape((1, 300))
        count += 1.
    if count != 0:
        vec /= count
    _arrays[0,:] = vec
    vectorized_array = pd.DataFrame(_arrays)
    pred = model2.predict([vectorized_array.iloc[:,0:300]])
    value = np.argmax(pred,axis=-1)
    labels = ['adventure','art and music','food','history','manufacturing','nature','science and technology','sports','travel']
    category = labels[value.item()]
    result = tk.Label(frame, text= str('Caption:') + str(label) + '\n' + str('Predicted Label :') + str(category)).pack()


def recommend():
    captions = pd.read_csv('captions.csv')
    is_category = captions['Category'].str.lower()== category
    df = captions.loc[is_category]
    df.columns = ('Category', 'Caption')
    df.head()
    caption_list = df['Caption']
    result = tk.Label(frame, text = str('Recommended Captions for the label:') + str('\n')).pack()
    count = 1
    for caption in caption_list:
        result = tk.Label(frame, text = str(count) + str('.') + str(caption)).pack()
        count = count+1


root = tk.Tk()
root.title('Image Captioning')
root.iconbitmap('class.ico')
root.resizable(False, False)
tit = tk.Label(root, text="Portable Image Classifier", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=700, width=1100, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="black", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Caption Image',
                        padx=35, pady=10,
                        fg="black", bg="grey", command=classify)

class_image.pack(side=tk.RIGHT)
recommend_caption = tk.Button(root, text = 'Recommend Captions', padx=35, pady=10,
                             fg = "black", bg = "grey", command=recommend)
recommend_caption.pack(side=tk.RIGHT)
#vgg_model = vgg16.VGG16(weights='imagenet')
model1 = tensorflow.keras.models.load_model('OUTPUT/saved_model.hp5')
model2 = tensorflow.keras.models.load_model('model_text_categorize.h5')
#weights = tensorflow.keras.models.load_model('model.h5')
#model= moodel
root.mainloop()


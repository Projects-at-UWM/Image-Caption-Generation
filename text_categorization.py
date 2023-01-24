import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from gensim.models import Word2Vec,KeyedVectors
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt


data = pd.read_csv('merged_data_file.csv')
data.head(5)

data.shape

del data['Unnamed: 0']
del data['Unnamed: 0.1']
del data['file']
data.isnull().sum()
df = data[data['Description'].notnull()]
df.head()

df = df.drop_duplicates(subset = 'Video Id')
boolean = df['Video Id'].duplicated().any()
print(boolean)
df.set_index('Video Id')
df.head()

df['Category'] = df['Category'].str.replace(r'\d+','')
df['Description'] = df['Description'].str.replace(r'\d+','')
df['Title'] = df['Title'].str.replace(r'\d+','')
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]
for char in spec_chars:
    df['Title'] = df['Title'].str.replace(char, ' ')
    df['Description'] = df['Description'].str.replace(char, ' ')
    df['Category'] = df['Category'].str.replace(char, ' ')

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
for i, row in df.iterrows():
    text_t = remove_emoji(row['Title'])
    text_d = remove_emoji(row['Description'])
    df.at[i,'Title']= text_t
    df.at[i,'Description'] = text_d

for i, row in df.iterrows():
    text_t = ' '.join([w for w in row['Title'].split() if wordnet.synsets(w)])
    df.at[i,'Title'] = text_t
    text_d = ' '.join([w for w in row['Description'].split() if wordnet.synsets(w)])
    df.at[i,'Description'] = text_d

for i, row in df.iterrows():
    word_list_t = row['Title'].lower().split() 
    filtered_words = [word for word in word_list_t if word not in stopwords.words('english')]
    text_t = ' '.join(filtered_words)
    df.at[i,'Title']= text_t
    word_list_d = row['Description'].lower().split() 
    filtered_words = [word for word in word_list_d if word not in stopwords.words('english')]
    text_d = ' '.join(filtered_words)
    df.at[i,'Description'] = text_d

df.Category.value_counts()

# Tokenization
df['descriptions_token'] = df['Description'].apply(lambda x: x.split())

def token_check(x,model):
  """
  1.Check if the token exists in the word2vec model vocab. 
  2.Check if the length of the token is greater than 3 
  """
  token_list=[]
  for i in x:
      if len(i) > 3 and i in model.vocab:
          token_list.append(i)
      else:
          continue
  return token_list

def word_vector(tokens,size,model):

  """Averaging the word vectors"""
  
  vec = np.zeros(size).reshape((1, size))
  count = 0
  for word in tokens:
    vec += model[word].reshape((1, size))
    count += 1.
  if count != 0:
      vec /= count
  return vec

#Load the word2vec pre trained Model to get the word embeddings for each token
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True,limit=1000000)

df['descriptions_token'] = df['descriptions_token'].apply(lambda x:token_check(x,model))

df.reset_index(drop=True, inplace=True)

wordvec_arrays = np.zeros((len(df.descriptions_token),300))

for i in range(len(df.descriptions_token)):
  wordvec_arrays[i,:] = word_vector(df.descriptions_token[i],300,model)

vectorized_df = pd.DataFrame(wordvec_arrays)

target_one_hot_encoded = pd.get_dummies(df['Category'])

target_one_hot_encoded.head()

dataset = pd.merge(vectorized_df,target_one_hot_encoded,left_index=True, right_index=True)

dataset_shuffled = dataset.reindex(np.random.permutation(dataset.index))

#splitting of data

target=['adventure','art and music','food','history','manufacturing','nature','science and technology','sports','travel']
X = dataset_shuffled.loc[:,~dataset_shuffled.columns.isin(target)]
y = dataset_shuffled.loc[:,dataset_shuffled.columns.isin(target)]

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75,test_size=0.25,random_state=101)

def NN_arch1(lrate=0.0001):
    model = models.Sequential()
    model.add(layers.Dense(300,input_dim = 300, activation='relu'))
    model.add(layers.Dense(5,activation='relu'))
    model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(9,activation='softmax'))
    opt = keras.optimizers.Adam(lr=lrate)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

def NN_arch2(lrate=0.0001):
    model = models.Sequential()
    model.add(layers.Dense(300,input_dim = 300, activation='relu'))
    model.add(layers.Dense(5,activation='sigmoid'))
    model.add(Dropout(0.20))
    model.add(layers.Dense(6,activation='relu'))
    model.add(Dropout(0.10))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(9,activation='softmax'))
    opt = keras.optimizers.Adam(lr=lrate)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

def model_fit(model,X_train,y_train,X_test,y_test,epoch_val=50):
    callbacks = EarlyStopping(monitor='val_loss',mode='min',patience=3)
    model.fit(X_train, y_train, epochs=epoch_val,batch_size=50)
    val_loss, val_acc = model.evaluate(X_test,y_test)
    print("val_loss, val_acc",val_loss, val_acc)
    return val_loss,val_acc

def history_plot(history):
    """ history plot """
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    training_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.figure(figsize=(5,3))

    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    #Visualize accuracy history
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, test_acc, 'b-')
    plt.legend(['Training acc', 'Test acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.show();

model_NN = NN_arch2(0.0009)
#val_loss,val_acc = model_fit(model_NN,100)
history = model_NN.fit(X_train,y_train,epochs=100,verbose=1,batch_size=100,validation_data=(X_test, y_test)) 
history_plot(history)

model_NN.save('model_text_categorize.h5')
#Prediction
loaded_model = tf.keras.models.load_model('model_text_categorize.h5')
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True,limit=1000000)

def clean_description_text(description):
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
  return text

description = 'robot'
clean_description = clean_description_text(description)
description_tokens = list(clean_description.split(" "))
description_tokens_filtered = token_check(description_tokens,model)

_arrays = np.zeros((1, 300))
_arrays[0,:] = word_vector(description_tokens_filtered,300,model)
vectorized_array = pd.DataFrame(_arrays)

pred = loaded_model.predict([vectorized_array.iloc[:,0:300]])

value = np.argmax(pred,axis=-1)
labels = ['adventure','art and music','food','history','manufacturing','nature','science and technology','sports','travel']

print("Predicted Label : " , labels[value.item()])  
vectorized_df.head()
vectorized_df_lgbm = vectorized_df
vectorized_df_lgbm['Target'] = df.Category

import pandas
from sklearn import preprocessing 

def target_label_encoding(x):
  labels = ['adventure','art and music','food','history','manufacturing','nature','science and technology','sports','travel']
  if x == labels[0]:
    return 0
  elif x == labels[1]:
    return 1
  elif x == labels[2]:
    return 2
  elif x == labels[3]:
    return 3
  elif x == labels[4]:
    return 4
  elif x == labels[5]:
    return 5
  elif x == labels[6]:
    return 6
  elif x == labels[7]:
    return 7
  else:
    return 8

vectorized_df_lgbm.Target = vectorized_df_lgbm.Target.apply(lambda x : target_label_encoding(x))
dataset_shuffled = vectorized_df.reindex(np.random.permutation(vectorized_df.index))

from sklearn.model_selection import train_test_split

#target=['adventure','art and music','food','history','manufacturing','nature','science and technology','sports','travel']
X = dataset_shuffled.loc[:,~dataset_shuffled.columns.isin(['Target'])]
y = dataset_shuffled.loc[:,dataset_shuffled.columns.isin(['Target'])]

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.75,test_size=0.25,random_state=101)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

estimator = lgb.LGBMClassifier(boosting_type='gbdt', 
                               learning_rate = 0.075, 
                               metric = 'multiclass', n_estimators = 20,
                               num_leaves = 38)

param_grid = {
    'n_estimators': [x for x in [75,100]],
    'learning_rate': [0.01,0.1,0.2],
    'num_leaves': [5,10],
    'boosting_type' : ['gbdt'],
    'objective' : ['multiclass'],
    'metric': ['multiclass'],
    'lambda_l1' : [0.10],
    'lambda_l2' : [0.20],
    'random_state' : [501]}

gridsearch = GridSearchCV(estimator, param_grid)

gridsearch.fit(X_train, y_train.values.ravel(),eval_set = [(X_test, y_test)],eval_metric = ['multi_logloss'],early_stopping_rounds = 10)

print('Best parameters found by grid search are:', gridsearch.best_params_)

gbm = lgb.LGBMClassifier(boosting_type= 'gbdt', learning_rate=0.05, 
                         n_estimators=400, num_leaves= 10,
                         lambda_l1=0.5,
                         lambda_l2=0.4,
                         objective= 'multiclass', random_state= 501)

gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['multiclass'],
early_stopping_rounds=20)


# Test Metric Validation
y_pred_prob = gbm.predict_proba(X_test)
pred_list=[]
for i in range(len(y_pred_prob)):
  pred_list.append(np.argmax(y_pred_prob[i]))

test_predictions  = pd.DataFrame({'prediction' : pred_list,'Actual':y_test.Target})

from sklearn import metrics
print(metrics.accuracy_score(test_predictions.prediction, test_predictions.Actual))
print(gbm.score(X_train,y_train))
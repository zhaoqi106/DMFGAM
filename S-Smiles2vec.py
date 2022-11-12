import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GRU, LSTM, Dense, Activation, Dropout, MaxPooling1D, SpatialDropout1D
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
import warnings
warnings.filterwarnings('ignore')

# In[3]:


df = pd.read_csv(r"D:\科研\王天一程序\data\newcadata.csv")

smiles = df['smiles']
targets = df['labels']
# In[18]:

# smiles_train, smiles_test, targets_train, targets_test = train_test_split(smiles, targets, test_size=0.2)
smiles_train = smiles[0:12620]
smiles_test = smiles[-740:]

targets_train = targets[0:12620]
targets_test = targets[-740:]

# In[26]:


batch_size = 100
tokenizer = Tokenizer(filters='', lower=False, char_level=True)
tokenizer.fit_on_texts(smiles.values)
one_hot_train = tokenizer.texts_to_sequences(smiles_train.values)
one_hot_test = tokenizer.texts_to_sequences(smiles_test.values)
one_hot_train = pad_sequences(one_hot_train, padding='post', maxlen=100)
one_hot_test = pad_sequences(one_hot_test, padding='post', maxlen=100)


# In[33]:


model = Sequential()
model.add(Embedding(len(tokenizer.index_docs) + 1, 50, input_length=one_hot_train.shape[1]))
model.add(Conv1D(filters=192, kernel_size=5, activation='relu'))
model.add(SpatialDropout1D(0.2))
model.add(MaxPooling1D(4))
model.add(LSTM(units=100, return_sequences=True, activation='relu'))
model.add(LSTM(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.999)
from sklearn.metrics import roc_auc_score
from keras import backend as K
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N


def AUC(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

def metric(y_true, y_pred):

        y_pred_positive = K.round(K.clip(y_pred, 0, 1))
        y_pred_negative = 1 - y_pred_positive

        y_positive = K.round(K.clip(y_true, 0, 1))
        y_negative = 1 - y_positive

        TP = K.sum(y_positive * y_pred_positive)
        TN = K.sum(y_negative * y_pred_negative)

        FP = K.sum(y_negative * y_pred_positive)
        FN = K.sum(y_positive * y_pred_negative)

        return TP, TN, FP, FN


def SPE(y_true, y_pred):
    TP, TN, FP, FN = metric(y_true, y_pred)
    return TN / (TN + FP)

def SEN(y_true, y_pred):
    TP, TN, FP, FN = metric(y_true, y_pred)
    return TP / (TP + FN)

def NPV(y_true, y_pred):
    TP, TN, FP, FN = metric(y_true, y_pred)
    return TN / (TN + FN)

def PPV(y_true, y_pred):
    TP, TN, FP, FN = metric(y_true, y_pred)
    return TP / (TP + FP)

def MCC(y_true, y_pred):
    TP, TN, FP, FN = metric(y_true, y_pred)
    return (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5



model.compile(optimizer='rmsprop', loss='binary_crossentropy' ,metrics=['accuracy', AUC, SPE, SEN, NPV, PPV, MCC])
model.summary()


# In[29]:


tensorboardCallback = TensorBoard()
model.fit(one_hot_train, targets_train, epochs=20, validation_split=0.2, callbacks=[tensorboardCallback])
score = model.evaluate(one_hot_test, targets_test)
print(score[1:])

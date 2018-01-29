## Load Library
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 150
VALIDATION_SPLIT=0.2
Train_Len=38932
data = pd.read_csv("clean_data.csv")
texts = data.Description[0:Train_Len-1]
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

labels = to_categorical(LabelEncoder().fit_transform(data.Is_Response[0:Train_Len-1]))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64,kernel_regularizer=regularizers.l2(0.01))))
#model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))
for _ in range(0,5):
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train,batch_size=32,epochs=2,verbose=2)
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    print(acc)

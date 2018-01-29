## Load Library
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
#from matplotlib import pyplot
import pandas as pd
from keras.utils import to_categorical
from keras import regularizers


## Prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
	# create the tokenizer
	tokenizer = Tokenizer(num_words= 20000)
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
	return Xtrain, Xtest

## Evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
	scores = list()
	n_repeats = 1
	n_words = Xtest.shape[1]
	for i in range(n_repeats):
		# define network
		model = Sequential()
		model.add(Dense(150, input_shape=(n_words,), activation='relu',
                                kernel_regularizer=regularizers.l2(0.01)))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(2, activation='softmax'))
		# compile network
		model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
		# fit network
		model.fit(Xtrain, ytrain, epochs=10, batch_size=128, verbose=2)
		# evaluate
		loss, acc = model.evaluate(Xtest, ytest, verbose=0)
		scores.append(acc)
		print('%d accuracy: %s' % ((i+1), acc))
	return scores# evaluate a neural network model

data = pd.read_csv("clean_data.csv")
train_docs = data.Description[0:35931]
test_docs = data.Description[35932:38931]
data.Is_Response[0:38931] = (LabelEncoder()).fit_transform(data.Is_Response[0:38931])
ytrain = to_categorical(data.Is_Response[0:35931])
ytest = to_categorical(data.Is_Response[35932:38931])

modes = [ 'freq']
results = DataFrame()
for mode in modes:
	# prepare data for mode
	Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
	# evaluate model on data for mode
	results[mode] = evaluate_mode(Xtrain, ytrain, Xtest, ytest)
# summarize results
print(results.describe())
# plot results
#results.boxplot()
#pyplot.show()


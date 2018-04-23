import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import tensorflow as tf
import random as rn
from keras.layers import Dense,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split


# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as bK

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
bK.set_session(sess)



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings



angles = np.loadtxt('a_1.csv', delimiter=" ")
print(angles.shape)
#print(angles)

positions = np.loadtxt('iX_1.csv', delimiter=" ")
print(positions.shape)
#print(positions)

data_set=np.concatenate((angles,positions), axis=1)
#print(data_set)

X_training, X_test, Y_training, Y_test =train_test_split(angles, positions, test_size= 0.2, random_state=42)

batch_size=32

# create model
model = Sequential()
model.add(Dense(100, input_dim=4, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))

#compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#fit model
history=model.fit(X_training, Y_training, batch_size= batch_size, epochs=50, verbose=0, validation_split=0.1)
#validation_split=0

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



############################
# evaluate the model
scores = model.evaluate(X_test, Y_test,batch_size=batch_size)
print("\nTest loss: " , scores[0])
print("\nTest accuracy: %.2f%%" %  (scores[1]*100))

predictions=model.predict(X_test,batch_size=batch_size)
v=abs(predictions-Y_test)
print(v)
err=np.linalg.norm(v,2)
print("\n norma-2 errore: ",err)

######################

plt.scatter(Y_test[:,0],Y_test[:,1])
plt.scatter(predictions[:,0],predictions[:,1], c='r', marker='*')
plt.show()

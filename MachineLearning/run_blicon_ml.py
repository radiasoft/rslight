import numpy as np
import h5py as h5
import os 
import scipy as scp
import sklearn as skl
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from keras.models import Sequential, Model
from keras.layers import Dense, GaussianNoise, Input, Conv2D, Flatten
import argparse
from mlhelpers4 import PlotLosses, plot_a_bunch_of_beams,make_dataset
plot_losses=PlotLosses()

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=100)
parser.add_argument('--batch_size', default=100)
parser.add_argument('--weights')
parser.add_argument('--pretrain_string', default=None)
parser.add_argument('--model', default = 'Jon', choices = ['Jon', '1'])
args = parser.parse_args()



############### Data ############### 
inputs = np.load('image_inputs.npy')
outputs = np.load('parameter_outputs.npy')
#scale the inputs and outputs
transformer_y = RobustScaler().fit(outputs)
Y_scaled = transformer_y.transform(outputs)

# scale the inputs to be on a range of pm 0.8
inputs_s =  1.6 * (inputs / np.max(inputs))- 0.8

split = 70
x_train, x_val, y_train, y_val = train_test_split(inputs_s, Y_scaled, 
                                                test_size = (100 - split) / 100.,
                                                random_state = 42) 

x_train = x_train.reshape(x_train.shape[0],80,80,1)
x_val = x_val.reshape(x_val.shape[0],80,80,1)

############### Model ############### 

#create model
model = Sequential()

if args.model == 'Jon':
    model.add(Conv2D(10, kernel_size=5, input_shape=(80,80,1)))
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(10, kernel_size=3, input_shape=(80,80,1)))
    model.add(GaussianNoise(0.1))
elif args.model =='1':
    model.add(Conv2D(1,kernel_size=3, input_shape=(80,80,1)))
    
model.add(Flatten())
model.add(Dense(2))
    
folder = 'Results/' + str(args.model) + '/n_epochs' + str(args.n_epochs) + '_batchsize' + str(args.batch_size) 
if args.pretrain_string is not None:
    folder += '_' + str(args.pretrain_string)
folder = folder + '/'

    
if not os.path.isdir(folder):
    os.makedirs(folder)
    
    
model.compile(optimizer='adam', loss='mean_squared_error')

hist = model.fit(x=x_train, y=y_train, 
                validation_data= (x_val, y_val),
                 batch_size = int(args.batch_size),
                 shuffle = 'true',
                 epochs = int(args.n_epochs),
                 verbose = 'true',
                 callbacks = [plot_losses])


############### Eval ############### 

plt.figure()
plt.loglog(hist.epoch, hist.history['val_loss'])
plt.loglog(hist.epoch, hist.history['loss'])
plt.savefig(folder + 'logloss.png')

plt.figure()
plt.plot(hist.epoch, hist.history['val_loss'])
plt.plot(hist.epoch, hist.history['loss'])
plt.savefig(folder + 'loss.png')


pred_outputs = model.predict(x_val)


correlation_matrix_dz = np.corrcoef(pred_outputs[:,0], y_val[:,0])
correlation_xy_dz = correlation_matrix_dz[0,1]
r_squared_dz = correlation_xy_dz**2

correlation_matrix_dy = np.corrcoef(pred_outputs[:,1], y_val[:,1])
correlation_xy_dy = correlation_matrix_dy[0,1]
r_squared_dy = correlation_xy_dy**2


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.title('R^2: ' + str(r_squared_dz))
plt.hexbin(pred_outputs[:,0], y_val[:,0], cmap = 'bone_r')
plt.ylabel('dz validation')
plt.xlabel('dz predicted')
plt.tight_layout()
plt.subplot(1,2,2)
plt.title('R^2: ' + str(r_squared_dy))
plt.hexbin(pred_outputs[:,1], y_val[:,1], cmap = 'bone_r')
plt.ylabel('dx validation')
plt.xlabel('dx predicted')
plt.tight_layout()
plt.savefig(folder + 'cnn_performance.pdf')
plt.show()
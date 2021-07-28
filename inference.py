import numpy as np
from tensorflow import keras

# Read dataset
X_train = np.rollaxis(np.load('dataset/X_train.npz')['arr_0'], 2, 1) # channel_first to channel_last
y_train = np.load('dataset/y_train.npz')['arr_0']

X_test = np.rollaxis(np.load('dataset/X_test.npz')['arr_0'], 2, 1) # channel_first to channel_last
y_test = np.load('dataset/y_test.npz')['arr_0']
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Sensor and length selections
sensor_selectors = [
    [0, 1, 2, 3, 4, 5, 6, 7],   # SG + Acc. + Gyro.
    [0, 1],                     # SG
    [2, 3, 4],                  # Acc.
    [5, 6, 7],                  # Gyro.
]

sample_selectors = [
    [350, 995], # 645
    [350, 950], # 600
    [350, 900], # 550
    [350, 850], # 500
    [350, 800], # 450
]

# Element-wise Inference 
sensor_selector, sample_selector = sensor_selectors[0], sample_selectors[0] # SG + Acc. + Gyro., from 350 to 995
golfresnet = keras.models.load_model(
    'pretrained/golfresnet_sensor_{}_sample_{}.hdf5'.format(str(sensor_selector), str(sample_selector))) # load pretrained model
y_pred = [] # predictions
for x in X_test[..., sample_selector[0]:sample_selector[1], sensor_selector]:
    y_pred += [golfresnet.predict(x[np.newaxis]).argmax()] # predict() outputs logits, argmax() gets prediction
test_acc = np.mean(y_pred == y_test) # average of test accuracy
print('test_acc: ', test_acc)

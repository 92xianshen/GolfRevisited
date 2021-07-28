# Please note that it is NOT recommended to use directly

import numpy as np
from tensorflow import keras
from model.GolfResNet import define_golfresnet

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

for sensor_selector in sensor_selectors:
    for sample_selector in sample_selectors:
        print(X_train[..., sample_selector[0]:sample_selector[1], sensor_selector].shape[1:3])
        golfresnet = define_golfresnet(input_shape=X_train[..., sample_selector[0]:sample_selector[1], sensor_selector].shape[1:3])
        
        golfresnet.fit(X_train[..., sample_selector[0]:sample_selector[1], sensor_selector], y_train, epochs=100)
        
        golfresnet.save('golfresnet_checkpoints/golfresnet_sensor_{}_sample_{}.hdf5'.format(str(sensor_selector), str(sample_selector)))
        
        test_loss, test_acc = golfresnet.evaluate(X_test[..., sample_selector[0]:sample_selector[1], sensor_selector], y_test)
        print('golfresnet_sensor_{}_sample_{}'.format(str(sensor_selector), str(sample_selector)), 'test acc: ', test_acc)
from tensorflow import keras

def define_golfresnet(input_shape=[995, 8]):

    l_in = keras.layers.Input(shape=input_shape, name='l_in')

    l_conv1 = keras.layers.Conv1D(filters=28, kernel_size=[3, ], strides=[1, ], 
                                  padding='valid', activation=keras.activations.relu)(l_in)

    l_maxpool1 = keras.layers.MaxPool1D(pool_size=[2, ])(l_conv1)

    # blockstack 1
    l_bs_1_conv1 = keras.layers.Conv1D(filters=28, kernel_size=[3, ], strides=[1, ], 
                                       padding='same', activation=keras.activations.relu)(l_maxpool1)

    l_bs_1_conv2 = keras.layers.Conv1D(filters=28, kernel_size=[3, ], strides=[1, ],
                                      padding='same', activation=keras.activations.relu)(l_bs_1_conv1)

    l_bs_1_add = keras.layers.Add()([l_maxpool1, l_bs_1_conv2])

    # resblock 1
    l_id_1_conv1 = keras.layers.Conv1D(filters=56, kernel_size=[3, ], strides=[2, ], 
                                      padding='same', activation=keras.activations.relu)(l_bs_1_add)

    l_id_1_conv2 = keras.layers.Conv1D(filters=56, kernel_size=[3, ], strides=[1, ], 
                                      padding='same', activation=keras.activations.relu)(l_id_1_conv1)

    l_id_1_proj = keras.layers.Conv1D(filters=56, kernel_size=[1, ], strides=[2, ], 
                                     padding='same', activation=keras.activations.relu)(l_bs_1_add)

    l_id_1_add = keras.layers.Add()([l_id_1_conv2, l_id_1_proj])

    l_maxpool2 = keras.layers.MaxPool1D(pool_size=[2, ])(l_id_1_add)

    # blockstack 2

    l_bs_2_conv1 = keras.layers.Conv1D(filters=56, kernel_size=[3, ], strides=[1, ], 
                                      padding='same', activation=keras.activations.relu)(l_maxpool2)

    l_bs_2_conv2 = keras.layers.Conv1D(filters=56, kernel_size=[3, ], strides=[1, ], 
                                      padding='same', activation=keras.activations.relu)(l_bs_2_conv1)

    l_bs_2_add = keras.layers.Add()([l_maxpool2, l_bs_2_conv2])

    # resblock 2

    l_id_2_conv1 = keras.layers.Conv1D(filters=112, kernel_size=[3, ], strides=[2, ], 
                                      padding='same', activation=keras.activations.relu)(l_bs_2_add)

    l_id_2_conv2 = keras.layers.Conv1D(filters=112, kernel_size=[3, ], strides=[1, ], 
                                      padding='same', activation=keras.activations.relu)(l_id_2_conv1)

    l_id_2_proj = keras.layers.Conv1D(filters=112, kernel_size=[1, ], strides=[2, ], 
                                     padding='same', activation=keras.activations.relu)(l_bs_2_add)

    l_id_2_add = keras.layers.Add()([l_id_2_conv2, l_id_2_proj])

    l_maxpool3 = keras.layers.MaxPool1D(pool_size=[2, ])(l_id_2_add)

    # blockstack 3

    l_bs_3_conv1 = keras.layers.Conv1D(filters=112, kernel_size=[3, ], strides=[1, ], 
                                      padding='same', activation=keras.activations.relu)(l_maxpool3)

    l_bs_3_conv2 = keras.layers.Conv1D(filters=112, kernel_size=[3, ], strides=[1, ], 
                                      padding='same', activation=keras.activations.relu)(l_bs_3_conv1)

    l_bs_3_add = keras.layers.Add()([l_maxpool3, l_bs_3_conv2])

    # resblock 3

    l_id_3_conv1 = keras.layers.Conv1D(filters=224, kernel_size=[3, ], strides=[2, ], 
                                      padding='same', activation=keras.activations.relu)(l_bs_3_add)

    l_id_3_conv2 = keras.layers.Conv1D(filters=224, kernel_size=[3, ], strides=[1, ], 
                                      padding='same', activation=keras.activations.relu)(l_id_3_conv1)

    l_id_3_proj = keras.layers.Conv1D(filters=224, kernel_size=[1, ], strides=[2, ], 
                                     padding='same', activation=keras.activations.relu)(l_bs_3_add)

    l_id_3_add = keras.layers.Add()([l_id_3_conv2, l_id_3_proj])

    l_maxpool4 = keras.layers.MaxPool1D(pool_size=[2, ])(l_id_3_add)

    l_flat = keras.layers.Flatten(name='l_flat')(l_maxpool4)

    # fc 1

    l_dropout1 = keras.layers.Dropout(rate=0.5)(l_flat)

    l_fc1 = keras.layers.Dense(units=256, activation=keras.activations.relu)(l_dropout1)

    # fc 2

    l_dropout2 = keras.layers.Dropout(rate=0.5)(l_fc1)

    l_fc2 = keras.layers.Dense(units=19, activation=keras.activations.softmax)(l_dropout2)

    golfresnet = keras.models.Model(inputs=l_in, outputs=l_fc2)
    
    golfresnet.compile(optimizer=keras.optimizers.Adam(), 
                       loss=keras.losses.sparse_categorical_crossentropy, 
                      metrics=['accuracy'])
    
    return golfresnet


if __name__ == '__main__':
    golfresnet = define_golfresnet()
    golfresnet.summary()
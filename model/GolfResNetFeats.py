from tensorflow import keras
from GolfResNet import define_golfresnet

def define_golfresnet_feats(input_shape=[995, 8]):
    golfresnet = define_golfresnet(input_shape)
    golfresnet_feats = keras.models.Model(inputs=golfresnet.input, outputs=golfresnet.get_layer('l_flat').output)
    return golfresnet_feats


if __name__ == '__main__':
    golfresnet_feats = define_golfresnet_feats()
    golfresnet_feats.summary()
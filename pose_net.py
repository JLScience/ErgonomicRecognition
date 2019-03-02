# the network utilized to estimate the gestures (trained on COCO keypoints challenge)
# paper: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
# adapted from: https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/blob/new-generation/model.py

from keras.models import Model
from keras.layers import Input, Activation, Lambda, Conv2D, MaxPooling2D, Concatenate


# original weights from the paper converted from caffe:
WEIGTHS_PATH = "data/model.h5"


# --- core layers:

def relu(z):
    return Activation('relu')(z)


def conv(z, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(z)
    return x1


def pooling(z, ks, st, name):
    z = MaxPooling2D((ks, ks), strides=(st, st), name=name)(z)
    return z


# --- core building blocks:

def vgg_block(z):
    # Block 1
    z = conv(z, 64, 3, "conv1_1")
    z = relu(z)
    z = conv(z, 64, 3, "conv1_2")
    z = relu(z)
    z = pooling(z, 2, 2, "pool1_1")

    # Block 2
    z = conv(z, 128, 3, "conv2_1")
    z = relu(z)
    z = conv(z, 128, 3, "conv2_2")
    z = relu(z)
    z = pooling(z, 2, 2, "pool2_1")

    # Block 3
    z = conv(z, 256, 3, "conv3_1")
    z = relu(z)
    z = conv(z, 256, 3, "conv3_2")
    z = relu(z)
    z = conv(z, 256, 3, "conv3_3")
    z = relu(z)
    z = conv(z, 256, 3, "conv3_4")
    z = relu(z)
    z = pooling(z, 2, 2, "pool3_1")

    # Block 4
    z = conv(z, 512, 3, "conv4_1")
    z = relu(z)
    z = conv(z, 512, 3, "conv4_2")
    z = relu(z)

    # Additional non vgg layers
    z = conv(z, 256, 3, "conv4_3_CPM")
    z = relu(z)
    z = conv(z, 128, 3, "conv4_4_CPM")
    z = relu(z)

    return z


def stage1_block(z, num_p, branch):
    # Block 1
    z = conv(z, 128, 3, "conv5_1_CPM_L%d" % branch)
    z = relu(z)
    z = conv(z, 128, 3, "conv5_2_CPM_L%d" % branch)
    z = relu(z)
    z = conv(z, 128, 3, "conv5_3_CPM_L%d" % branch)
    z = relu(z)
    z = conv(z, 512, 1, "conv5_4_CPM_L%d" % branch)
    z = relu(z)
    z = conv(z, num_p, 1, "conv5_5_CPM_L%d" % branch)

    return z


def stageT_block(z, num_p, stage, branch):
    # Block 1
    z = conv(z, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
    z = relu(z)
    z = conv(z, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
    z = relu(z)
    z = conv(z, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
    z = relu(z)
    z = conv(z, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
    z = relu(z)
    z = conv(z, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
    z = relu(z)
    z = conv(z, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
    z = relu(z)
    z = conv(z, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))

    return z

# ---


def build_net():
    stages = 6
    np_branch1 = 38
    np_branch2 = 19
    input_shape = (None, None, 3)

    img_input = Input(shape=input_shape)
    img_normalized = Lambda(lambda z: z / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized)

    # stage 1
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2)
        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
    model.load_weights(WEIGTHS_PATH)

    print('Successfully loaded pose estimator with weights!')

    return model


if __name__ == '__main__':
    # build and show description of the network:
    pose_net = build_net()
    pose_net.summary()

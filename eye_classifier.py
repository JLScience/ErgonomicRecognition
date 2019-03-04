
import os
import numpy as np
import cv2 as cv
import h5py

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.optimizers import Adam


# called from main.py if eye data gathering is activated; an image of each eye is saved
def collect_data(l_eye, r_eye, eyes_open):
    path = 'data/eye_data/'
    path = path + 'open/' if eyes_open else path + 'closed/'
    counter = int(np.loadtxt(path + 'count.txt'))

    l_eye = cv.resize(l_eye, (64, 64))
    r_eye = cv.resize(r_eye, (64, 64))

    cv.imwrite(path + str(counter) + '.png', l_eye)
    cv.imwrite(path + str(counter+1) + '.png', r_eye)

    f = open(path + 'count.txt', mode='w')
    f.write(str(counter+2))
    f.close()
    print('Saved images {}.png and {}.png!'.format(counter, counter+1))


# create .hdf5 file containing two uint8 tensors of shape (num_images, image_size, image_size, image_channels)
def create_dataset():
    path = 'data/eye_data/'
    img_size = 64
    img_channels = 3
    labels = ['open', 'closed']
    f = h5py.File(path + 'dataset.hdf5')
    for label in labels:
        img_names = os.listdir(path + label)
        img_names.pop(img_names.index('count.txt'))
        num_labels = len(img_names)
        tensor = np.zeros((num_labels, img_size, img_size, img_channels), dtype=np.uint8)
        for idx, img_name in enumerate(img_names):
            img_path = path + label + '/' + img_name
            tensor[idx, :, :, :] = cv.imread(img_path)
        f.create_dataset(label, data=tensor)
        print('created dataset ' + label)


# load the dataset and split it into training and teseting data
def prepare_data(train_ratio=0.8):
    # load data:
    f = h5py.File('data/eye_data/dataset.hdf5')
    tensor_closed = np.array(f['closed'], dtype=np.uint8)
    tensor_opened = np.array(f['open'], dtype=np.uint8)
    num_samples = tensor_opened.shape[0]
    # shuffle data:
    p = np.random.permutation(num_samples)
    tensor_closed = tensor_closed[p]
    p = np.random.permutation(num_samples)
    tensor_opened = tensor_opened[p]
    # create training and test sets:
    split_idx = int(num_samples*train_ratio)
    x_train = np.append(tensor_opened[:split_idx, ...], tensor_closed[:split_idx, ...], axis=0)
    y_train = np.append(np.zeros(split_idx, dtype=np.uint8), np.ones(split_idx, dtype=np.uint8), axis=0)
    x_test = np.append(tensor_opened[split_idx:, ...], tensor_closed[split_idx:, ...], axis=0)
    y_test = np.append(np.zeros(num_samples - split_idx, dtype=np.uint8),
                       np.ones(num_samples - split_idx, dtype=np.uint8), axis=0)
    # shuffle training set:
    p = np.random.permutation(x_train.shape[0])
    x_train = x_train[p]
    y_train = y_train[p]
    return x_train, y_train, x_test, y_test


# augmentation:
def reflect_vertical(tensor):
    reflected = np.zeros(tensor.shape, dtype=tensor.dtype)
    rows = tensor.shape[1]
    cols = tensor.shape[2]
    for i in range(rows):
        for j in range(cols):
            reflected[:, i, j, :] = tensor[:, i, cols - 1 - j, :]
    return reflected


class Eye_Classifier():

    def __init__(self, apply=True):
        self.classifier = self.create_simple_convolutional_network()
        if apply:
            self.classifier.load_weights('data/eye_data/eye_classifier_weights.hdf5')
            print('Successfully loaded eye classifier with weights!')
        else:
            self.classifier.summary()

    def create_simple_convolutional_network(self):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same',
                         input_shape=(64, 64, 3)))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(2, activation='sigmoid'))
        return model

    def train(self):
        # fetch training and test data:
        x_train, y_train, x_test, y_test = prepare_data()

        # augment training data:
        x_train = np.append(x_train, reflect_vertical(x_train), axis=0)
        y_train = np.append(y_train, y_train, axis=0)
        p = np.random.permutation(x_train.shape[0])
        x_train = x_train[p]
        y_train = y_train[p]

        # pre-process data (input into range [-1, 1]):
        x_train = np.array(x_train / 127.5 - 1, dtype=np.float32)
        x_test = np.array(x_test / 127.5 - 1, dtype=np.float32)
        y_train = to_categorical(y_train, 2)

        # train classifier:
        opt = Adam()
        self.classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.classifier.fit(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, to_categorical(y_test, 2)), verbose=2)

        loss, acc = self.classifier.evaluate(x_test, to_categorical(y_test, 2))
        print('loss: ' + str(loss))
        print('acc:  ' + str(acc))

        self.classifier.save_weights('data/eye_data/eye_classifier_weights.hdf5')

        y_pred = self.classifier.predict(x_test)
        print(y_test)
        print(np.round(y_pred, 2))

    def apply(self, imgs):
        imgs = np.array(imgs / 127.5 - 1, dtype=np.float32)
        y = self.classifier.predict(imgs)
        return np.argmax(y, axis=1)


if __name__ == '__main__':
    # create_dataset()
    classifier = Eye_Classifier(apply=False)
    classifier.train()


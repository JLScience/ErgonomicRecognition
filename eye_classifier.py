
import os
import numpy as np
import cv2 as cv
import h5py

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout


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


class Classifier():

    def __init__(self):
        pass

    def create_classifier(self):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    create_dataset()
    # classifier = Classifier()
    # classifier.train()

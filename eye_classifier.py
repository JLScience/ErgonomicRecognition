
import numpy as np
import cv2 as cv


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

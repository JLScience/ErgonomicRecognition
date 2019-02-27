# The main file of the project containing the program flow

import numpy as np
import cv2 as cv

FPS = 30


def process_input_and_modify_screen(frame):

    # apply mirroring:
    frame = np.array(frame[:, ::-1, :])

    return frame


def main():
    # access camera:
    video = cv.VideoCapture(0)

    # repeat main loop until ESC button is pressed:
    run = True
    while run:
        # get next frame from webcam:
        ret, frame = video.read()

        # all calculations and modifications of the input and screen:
        frame = process_input_and_modify_screen(frame)

        # show image and wait for input:
        cv.imshow('ErgonomicRecognition', frame)
        ms_wait = int(1000 / FPS)
        key = cv.waitKey(ms_wait)
        if key == 27:
            break


if __name__ == '__main__':
    main()

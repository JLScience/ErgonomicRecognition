
import numpy as np
import cv2 as cv

FPS = 30

def main():
    # access camera:
    video = cv.VideoCapture(0)

    # repeat main loop until ESC button is pressed:
    run = True
    while run:
        # get next frame from webcam:
        ret, frame = video.read()

        # apply mirroring:
        frame = np.array(frame[:, ::-1, :])

        # show image and wait for input:
        cv.imshow('frame', frame)
        ms_wait = int(1000 / FPS)
        key = cv.waitKey(ms_wait)
        if key == 27:
            break


if __name__ == '__main__':
    main()

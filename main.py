# The main file of the project containing the program flow

import cv2 as cv

import engine
import painter

FPS = 30


def process_input_and_modify_screen(frame):

    # apply mirroring:
    frame = engine.mirror_vertical(frame)

    # calculate keypoints:
    key_points, max_scores = engine.calc_keypoints(frame)

    # plot keypoints:
    painter.paint_keypoints(frame, key_points, max_scores, 0.1, only_torso=False)

    # find face:
    face_mean_x, face_mean_y, face_size_v, face_size_h, face_angle = engine.find_face(key_points)
    painter.paint_point(frame, face_mean_x, face_mean_y)
    painter.paint_face_edge(frame, face_mean_x, face_mean_y, face_size_v, face_size_h, face_angle)

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

# The main file of the project containing the program flow

import cv2 as cv

import engine
import painter

FPS = 30


def process_input_and_modify_screen(frame, viola_jones=False, debug=True):

    # apply mirroring:
    frame = engine.mirror_vertical(frame)

    # calculate keypoints from posenet output:
    key_points, max_scores = engine.calc_keypoints(frame)

    # use Viola-Jones algorithm to find faces and eyes:
    if viola_jones:
        faces = engine.find_face_viola_jones(frame)
        eyes = []
        for x, y, w, h in faces:
            eyes.append(engine.find_eyes_viola_jones(frame[y:y+h, x:x+w]))

    if debug:
        # plot keypoints:
        painter.paint_keypoints(frame, key_points, max_scores, 0.1, only_torso=False)

        # find and plot face:
        face_mean_x, face_mean_y, face_size_v, face_size_h, face_angle = engine.find_face(key_points)
        painter.paint_point(frame, face_mean_x, face_mean_y)
        painter.paint_face_edge(frame, face_mean_x, face_mean_y, face_size_v, face_size_h, face_angle)

        # find and plot eye surrounding:
        l_eye, r_eye = engine.find_eyes(key_points)
        painter.paint_rect(frame, l_eye[0], l_eye[1], l_eye[2], l_eye[3])
        painter.paint_rect(frame, r_eye[0], r_eye[1], r_eye[2], r_eye[3])

        if viola_jones:
            for i, (x, y, w, h) in enumerate(faces):
                painter.paint_rect(frame, x, y, w, h)
                for ex, ey, ew, eh in eyes[i]:
                    painter.paint_rect(frame[y:y+h, x:x+w], ex, ey, ew, eh)



    # check if the posture is incorrect:
    posture_list = engine.check_posture(key_points, max_scores)

    # show signal if posture is incorrect:
    frame = painter.posture_alert(frame, posture_list)

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

# The main file of the project containing the program flow

import cv2 as cv

import engine
import painter
import eye_classifier

FPS = 30

# program switches:
RUN = True
DEBUG = False
APPLY_VIOLA_JONES = False
EYE_DATA_GATHERING = False
KEY_PRESSED = ""


def process_input_and_modify_screen(frame):

    # apply mirroring:
    frame = engine.mirror_vertical(frame)

    # calculate keypoints from posenet output:
    key_points, max_scores = engine.calc_keypoints(frame)

    # use Viola-Jones algorithm to find faces and eyes:
    if APPLY_VIOLA_JONES:
        faces = engine.find_face_viola_jones(frame)
        eyes = []
        for x, y, w, h in faces:
            eyes.append(engine.find_eyes_viola_jones(frame[y:y+h, x:x+w]))

    if DEBUG:
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

        # plot frames calculated by viola-jones algorithms:
        if APPLY_VIOLA_JONES:
            for i, (x, y, w, h) in enumerate(faces):
                painter.paint_rect(frame, x, y, w, h)
                for ex, ey, ew, eh in eyes[i]:
                    painter.paint_rect(frame[y:y+h, x:x+w], ex, ey, ew, eh)

    if EYE_DATA_GATHERING:
        # find and plot eye surrounding:
        l_eye, r_eye = engine.find_eyes(key_points, return_mode="")
        lx, ly, ldx, ldy = int(l_eye[0]), int(l_eye[1]), int(l_eye[2]), int(l_eye[3])
        rx, ry, rdx, rdy = int(r_eye[0]), int(r_eye[1]), int(r_eye[2]), int(r_eye[3])
        if KEY_PRESSED == 'c':
            eye_classifier.collect_data(frame[ly+1:ly+ldy, lx+1:lx+ldx], frame[ry+1:ry+rdy, rx+1:rx+rdx],
                                        eyes_open=False)
        if KEY_PRESSED == 'o':
            eye_classifier.collect_data(frame[ly+1:ly+ldy, lx+1:lx+ldx], frame[ry+1:ry+rdy, rx+1:rx+rdx],
                                        eyes_open=True)
        painter.paint_rect(frame, l_eye[0], l_eye[1], l_eye[2], l_eye[3])
        painter.paint_rect(frame, r_eye[0], r_eye[1], r_eye[2], r_eye[3])

    if RUN:
        # check if the posture is incorrect:
        posture_list = engine.check_posture(key_points, max_scores)

        # show signal if posture is incorrect:
        frame = painter.posture_alert(frame, posture_list)

        # check whether eyes are opened or closed:
        lx, ly, ldx, ldy, rx, ry, rdx, rdy = engine.find_eyes(key_points, return_mode='window')
        l_eye_open, r_eye_open = engine.eye_status(frame[ly+1:ly+ldy, lx+1:lx+ldx], frame[ry+1:ry+rdy, rx+1:rx+rdx])

        painter.paint_eye_status(frame, l_eye_open, r_eye_open)

    # plot key pressed:
    painter.paint_string(frame, "Key pressed: " + KEY_PRESSED, frame.shape[1] - 390, 225)

    return frame


def key_handling(key):
    global KEY_PRESSED
    if key == 27:
        exit(code=0)
    elif key == ord('c'):
        KEY_PRESSED = 'c'
    elif key == ord('o'):
        KEY_PRESSED = 'o'
    else:
        KEY_PRESSED = ""


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
        key_handling(key)


if __name__ == '__main__':
    main()

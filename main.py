# The main file of the project containing the program flow

import cv2 as cv

import engine
import painter
import eye_classifier

FPS = 60

# --- program switches:
RUN = True
START_SCREEN = True
ANALYSIS_SCREEN = False
INFO_SCREEN = False
DEBUG = False
APPLY_VIOLA_JONES = False
EYE_DATA_GATHERING = False


# --- global variables
KEY_PRESSED = ""
L_EYE_STATUS = []
R_EYE_STATUS = []
BLINK_TIMER = 0
CALIBRATING = False
CALIBRATING_HEAD_POSITIONS_Y = []
CALIBRATING_DISTANCE_NOSE_FACE_CENTER = []
HEAD_THRESHOLD_Y = -1


def process_input_and_modify_screen(frame):
    global L_EYE_STATUS, R_EYE_STATUS, BLINK_TIMER
    global CALIBRATING, CALIBRATING_HEAD_POSITIONS_Y, CALIBRATING_DISTANCE_NOSE_FACE_CENTER, HEAD_THRESHOLD_Y
    global START_SCREEN, ANALYSIS_SCREEN, INFO_SCREEN

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

    if EYE_DATA_GATHERING:
        # find and plot eye surrounding:
        l_eye, r_eye = engine.find_eyes(key_points, return_mode="")
        lx, ly, ldx, ldy = int(l_eye[0]), int(l_eye[1]), int(l_eye[2]), int(l_eye[3])
        rx, ry, rdx, rdy = int(r_eye[0]), int(r_eye[1]), int(r_eye[2]), int(r_eye[3])
        if KEY_PRESSED == 'c':
            eye_classifier.collect_data(frame[ly+1:ly+ldy, lx+1:lx+ldx], frame[ry+1:ry+rdy, rx+1:rx+rdx], eyes_open=False)
        if KEY_PRESSED == 'o':
            eye_classifier.collect_data(frame[ly+1:ly+ldy, lx+1:lx+ldx], frame[ry+1:ry+rdy, rx+1:rx+rdx], eyes_open=True)
        painter.paint_rect(frame, l_eye[0], l_eye[1], l_eye[2], l_eye[3])
        painter.paint_rect(frame, r_eye[0], r_eye[1], r_eye[2], r_eye[3])

    if RUN:
        if START_SCREEN:
            frame = painter.paint_signal_light(frame, mode='no_init')
            if KEY_PRESSED == 'a':
                ANALYSIS_SCREEN = True
                START_SCREEN = False

        elif ANALYSIS_SCREEN:

            frame = painter.paint_analysis_info(frame, CALIBRATING)

            # check whether eyes are opened or closed:
            lx, ly, ldx, ldy, rx, ry, rdx, rdy = engine.find_eyes(key_points, return_mode='window')
            l_eye_open, r_eye_open = engine.eye_status(frame[ly + 1:ly + ldy, lx + 1:lx + ldx],
                                                       frame[ry + 1:ry + rdy, rx + 1:rx + rdx])

            if DEBUG:
                painter.paint_eye_status(frame, l_eye_open, r_eye_open)
                painter.paint_string(frame, str(BLINK_TIMER), 20, 20)
                painter.paint_keypoints(frame, key_points, max_scores, threshold=0.1, only_torso=True)

            L_EYE_STATUS, R_EYE_STATUS = engine.track_eye_status(L_EYE_STATUS, R_EYE_STATUS, l_eye_open, r_eye_open)

            # check if calibration is started:
            if not CALIBRATING:
                CALIBRATING, BLINK_TIMER, L_EYE_STATUS, R_EYE_STATUS = engine.check_right_left_blink(L_EYE_STATUS, R_EYE_STATUS, BLINK_TIMER)

            if CALIBRATING:
                # store values for calibration:
                head_x, head_y = engine.get_mean_face_positions(key_points)
                distance = engine.calibration_get_mean_distance_nose_to_face_center(key_points)
                CALIBRATING_HEAD_POSITIONS_Y.append(head_y)
                CALIBRATING_DISTANCE_NOSE_FACE_CENTER.append(distance)

                # check if calibration is done:
                c_tmp, BLINK_TIMER, L_EYE_STATUS, R_EYE_STATUS = engine.check_right_left_blink(L_EYE_STATUS, R_EYE_STATUS, BLINK_TIMER)
                if c_tmp:
                    HEAD_THRESHOLD_Y = engine.calibration_get_threshold(CALIBRATING_HEAD_POSITIONS_Y, CALIBRATING_DISTANCE_NOSE_FACE_CENTER)
                    CALIBRATING_HEAD_POSITIONS_Y = []
                    CALIBRATING = False
                    ANALYSIS_SCREEN = False
                    INFO_SCREEN = True

                if DEBUG:
                    painter.paint_point(frame, head_x, head_y, color=(0, 0, 255))

        elif INFO_SCREEN:

            if DEBUG:
                # show threshold:
                painter.paint_line(frame, 0, HEAD_THRESHOLD_Y, frame.shape[1], HEAD_THRESHOLD_Y, color=(0, 0, 0))

            # check if the posture is incorrect:
            posture_list = engine.check_posture(key_points, max_scores, HEAD_THRESHOLD_Y)

            # show signal if posture is incorrect:
            frame = painter.posture_alert(frame, posture_list)

            # go back to analysis screen if 'a' is pressed:
            if KEY_PRESSED == 'a':
                INFO_SCREEN = False
                ANALYSIS_SCREEN = True

    return frame


def key_handling(key):
    global KEY_PRESSED
    if key == 27:
        exit(code=0)
    elif key == ord('c'):
        KEY_PRESSED = 'c'
    elif key == ord('o'):
        KEY_PRESSED = 'o'
    elif key == ord('a'):
        KEY_PRESSED = 'a'
    elif key == ord('d'):
        KEY_PRESSED = 'd'
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

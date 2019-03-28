# plotting, drawing and manipulating the screen

import numpy as np
import cv2 as cv

KEYPOINT_DESCRIPTIONS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip',
                         'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear', 'pt19']


# --- basic painting methods:

def paint_point(frame, x, y, r=5, color=(255, 255, 255)):
    cv.circle(frame, (int(x+0.5), int(y+0.5)), r, color, -1)


def paint_line(frame, x1, y1, x2, y2, color=(255, 255, 255)):
    cv.line(frame, (int(x1+0.5), int(y1+0.5)), (int(x2+0.5), int(y2+0.5)), color, 2)


def paint_rect(frame, x, y, w, h, color=(0, 0, 255), fill=1):
    cv.rectangle(frame, (int(x+0.5), int(y+0.5)), (int(x+w+0.5), int(y+h+0.5)), color, fill)


def paint_string(frame, string, x, y):
    cv.putText(frame, string, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv.LINE_AA)


# --- painting of the key points and face additions:

def paint_keypoints(frame, key_points, max_scores, threshold, only_torso=True):
    # showing only the keypoints of the upper body if only_torso is True
    num_bodyparts = key_points.shape[0]
    key_point_indizes = [0, 1, 2, 5, 14, 15, 16, 17] if only_torso else [i for i in range(num_bodyparts-1)]
    for i in key_point_indizes:
        if max_scores[i] > threshold:
            cv.circle(frame, (int(key_points[i, 1]+0.5), int(key_points[i, 0]+0.5)), 5, (0, 255, 0), -1)


def paint_face_edge(frame, c_x, c_y, face_size_v, face_size_h, angle):
    cv.ellipse(img=frame, center=(int(c_x+0.5), int(c_y+0.5)), axes=(int(face_size_v+0.5), int(face_size_h+0.5)),
               angle=angle, startAngle=0, endAngle=360, color=(255, 255, 255), thickness=2)


# --- painting of additional areas and information:

def paint_signal_light(frame, mode='no_init'):
    if mode == 'no_init':
        frame = 128*np.ones((200, 400, 3), dtype=np.uint8)
        paint_rect(frame, 10, 10, 80, 180, color=(0, 0, 0), fill=-1)
        paint_point(frame, 50, 55, 38, color=(255, 0, 0))
        paint_point(frame, 50, 145, 38, color=(255, 0, 0))
        paint_string(frame, 'Not calibrated!', 120, 80)
        paint_string(frame, 'Press A to start', 120, 130)
        return frame
    elif mode == 'good':
        paint_rect(frame, frame.shape[1]-250, 20, 100, 200, color=(0, 0, 0), fill=-1)
        paint_point(frame, frame.shape[1]-200, 70, 45, color=(128, 128, 128))
        paint_point(frame, frame.shape[1]-200, 170, 45, color=(0, 255, 0))
    elif mode == 'bad':
        paint_rect(frame, frame.shape[1] - 250, 20, 100, 200, color=(0, 0, 0), fill=-1)
        paint_point(frame, frame.shape[1] - 200, 70, 45, color=(0, 0, 255))
        paint_point(frame, frame.shape[1] - 200, 170, 45, color=(128, 128, 128))


def paint_analysis_info(frame, calibrating):
    new_frame = np.zeros((frame.shape[0] + 80, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
    new_frame[:frame.shape[0], :, :] = frame
    new_frame[frame.shape[0]:, :, :] = 255 * np.ones((80, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
    if not calibrating:
        paint_string(new_frame, 'Start calibration with right-left-blink', 20, new_frame.shape[0] - 50)
    else:
        paint_string(new_frame, 'Calibrating... finish with right-left-blink', 5, new_frame.shape[0] - 50)
    return new_frame


def posture_alert(frame, posture_list):
    notification_list = ['head twisted', 'head leaned forward', 'head leaned sidewards', 'head ducked']

    new_frame = np.zeros((frame.shape[0], frame.shape[1] + 400, frame.shape[2]), dtype=frame.dtype)
    new_frame[:, :frame.shape[1], :] = frame
    new_frame[:, frame.shape[1]:, :] = 255 * np.ones((frame.shape[0], 400, frame.shape[2]), dtype=frame.dtype)

    if True in posture_list:
        paint_signal_light(new_frame, mode='bad')
    else:
        paint_signal_light(new_frame, mode='good')

    if posture_list[0]:
        paint_string(new_frame, notification_list[0], frame.shape[1] + 10, 275)
    if posture_list[1]:
        paint_string(new_frame, notification_list[1], frame.shape[1] + 10, 325)
    if posture_list[2]:
        paint_string(new_frame, notification_list[2], frame.shape[1] + 10, 375)
    if posture_list[3]:
        paint_string(new_frame, notification_list[3], frame.shape[1] + 10, 425)

    return new_frame


def paint_eye_status(frame, l_eye_open, r_eye_open):
    # left eye:
    if l_eye_open == -1:
        paint_string(frame, 'left eye undefined', 20, frame.shape[0] - 20)
    elif l_eye_open < 0.5:
        paint_string(frame, 'left eye opened', 20, frame.shape[0] - 20)
    else:
        paint_string(frame, 'left eye closed', 20, frame.shape[0] - 20)
    # right eye:
    if r_eye_open == -1:
        paint_string(frame, 'right eye undefined', 320, frame.shape[0] - 20)
    elif r_eye_open < 0.5:
        paint_string(frame, 'right eye opened', 320, frame.shape[0] - 20)
    else:
        paint_string(frame, 'right eye closed', 320, frame.shape[0] - 20)

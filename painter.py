# plotting, drawing and manipulating the screen

import numpy as np
import cv2 as cv

KEYPOINT_DESCRIPTIONS = part_str = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip',
                                    'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear', 'pt19']


def paint_keypoints(frame, key_points, max_scores, threshold, only_torso=True):
    # showing only the keypoints of the upper body if only_torso is True
    num_bodyparts = key_points.shape[0]
    key_point_indizes = [0, 1, 2, 5, 14, 15, 16, 17] if only_torso else [i for i in range(num_bodyparts-1)]
    for i in key_point_indizes:
        if max_scores[i] > threshold:
            cv.circle(frame, (int(key_points[i, 1]+0.5), int(key_points[i, 0]+0.5)), 5, (0, 255, 0), -1)


def paint_point(frame, x, y, r=5, color=(255, 255, 255)):
    cv.circle(frame, (int(x+0.5), int(y+0.5)), r, color, -1)


def paint_rect(frame, x, y, w, h, color=(0, 0, 255)):
    cv.rectangle(frame, (int(x+0.5), int(y+0.5)), (int(x+w+0.5), int(y+h+0.5)), color, 1)


def paint_string(frame, string, x, y):
    cv.putText(frame, string, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)


def paint_face_edge(frame, c_x, c_y, face_size_v, face_size_h, angle):
    cv.ellipse(img=frame, center=(int(c_x+0.5), int(c_y+0.5)), axes=(int(face_size_v+0.5), int(face_size_h+0.5)),
               angle=angle, startAngle=0, endAngle=360, color=(255, 255, 255), thickness=2)


# TODO: embellish the information display
def posture_alert(frame, posture_list):
    notification_list = ['head twisted', 'head leaned forward', 'head leaned sidewards', 'head ducked']

    new_frame = np.zeros((frame.shape[0], frame.shape[1] + 400, frame.shape[2]), dtype=frame.dtype)
    new_frame[:, :frame.shape[1], :] = frame
    new_frame[:, frame.shape[1]:, :] = 255 * np.ones((frame.shape[0], 400, frame.shape[2]), dtype=frame.dtype)

    if posture_list[0]:
        paint_string(new_frame, notification_list[0], frame.shape[1] + 10, 25)
    if posture_list[1]:
        paint_string(new_frame, notification_list[1], frame.shape[1] + 10, 75)
    if posture_list[2]:
        paint_string(new_frame, notification_list[2], frame.shape[1] + 10, 125)
    if posture_list[3]:
        paint_string(new_frame, notification_list[3], frame.shape[1] + 10, 175)

    return new_frame

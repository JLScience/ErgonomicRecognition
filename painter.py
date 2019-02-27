# plotting, drawing and manipulating the screen

import cv2 as cv


def paint_keypoints(key_points, max_scores, threshold, frame, only_torso=True):
    # showing only the keypoints of the upper body if only_torso is True
    num_bodyparts = key_points.shape[0]
    key_point_indizes = [0, 1, 2, 5, 14, 15, 16, 17] if only_torso else [i for i in range(num_bodyparts-1)]
    for i in key_point_indizes:
        if max_scores[i] > threshold:
            cv.circle(frame, (int(key_points[i, 1]), int(key_points[i, 0])), 5, (0, 255, 0), -1)

# calculations on the networks output

import numpy as np
import cv2 as cv

import pose_net
import eye_classifier

POSE_NET = pose_net.build_net()
EYE_CLASSIFIER = eye_classifier.Eye_Classifier()
face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')


# --- basic functions:

def mirror_vertical(image):
    return np.array(image[:, ::-1, :])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def distance(p1, p2):
    return np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)


# --- functions for the application of the Viola-Jones method:

def find_face_viola_jones(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return faces


def find_eyes_viola_jones(face_frame):
    gray_frame = cv.cvtColor(face_frame, cv.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return eyes


# --- functions for key point calculation (using pose_net) and further analysis:

# calculate keypoints of highest confidence from the network output
def calc_keypoints(frame):
    heatmaps = POSE_NET.predict(np.expand_dims(frame, 0))
    scores = sigmoid(heatmaps[1])
    num_bodyparts = heatmaps[1].shape[3]
    heatmap_positions = np.zeros((num_bodyparts, 2))
    offset_vectors = np.zeros((num_bodyparts, 2))
    max_scores = np.zeros(num_bodyparts)
    for i in range(num_bodyparts):
        max_scores[i] = np.max(heatmaps[1][0, :, :, i])
        heatmap_positions[i, :] = np.unravel_index(np.argmax(scores[0, :, :, i]), (scores.shape[1], scores.shape[2]))
        offset_vectors[i, 0] = heatmaps[0][0, int(heatmap_positions[i, 0]), int(heatmap_positions[i, 1]), i]
        offset_vectors[i, 1] = heatmaps[0][
            0, int(heatmap_positions[i, 0]), int(heatmap_positions[i, 1]), i + num_bodyparts]
    key_points = heatmap_positions * 8 + offset_vectors
    return key_points, max_scores


# calculate face center dependant on the nose and eye positions
def get_mean_face_positions(key_points):
    face_mean_x = np.mean([key_points[0, 1], key_points[14, 1], key_points[15, 1]])
    face_mean_y = np.mean([key_points[0, 0], key_points[14, 0], key_points[15, 0]])
    return face_mean_x, face_mean_y


# calculate values required to plot an ellipse around the face
def find_face(key_points):
    x, y = get_mean_face_positions(key_points)
    w = 0.5 * distance(key_points[16], key_points[17])
    h = w * 1.5
    angle = 90 + 180 / np.pi * np.arctan2(key_points[15, 0] - key_points[14, 0], key_points[15, 1] - key_points[14, 1])
    return x, y, h, w, angle


# find a rectangle around each eye:
def find_eyes(key_points, return_mode):
    xm, ym = get_mean_face_positions(key_points)
    d = 0.5 * (distance([ym, xm], key_points[16]) + (distance([ym, xm], key_points[17]))) * 0.8
    l_eye = (key_points[14, 1] - d/2, key_points[14, 0] - d/2, d, d)
    r_eye = (key_points[15, 1] - d/2, key_points[15, 0] - d/2, d, d)
    if return_mode == 'window':
        lx, ly, ldx, ldy = int(l_eye[0]), int(l_eye[1]), int(l_eye[2]), int(l_eye[3])
        rx, ry, rdx, rdy = int(r_eye[0]), int(r_eye[1]), int(r_eye[2]), int(r_eye[3])
        return lx, ly, ldx, ldy, rx, ry, rdx, rdy
    else:
        return l_eye, r_eye


# calculate the mean distance from nose to face-center (required for threshold calibration):
def calibration_get_mean_distance_nose_to_face_center(key_points):
    nose_y = key_points[0, 0]
    face_mean_y = np.mean([key_points[0, 0], key_points[14, 0], key_points[15, 0]])
    return np.abs(nose_y - face_mean_y)


# calculate a threshold for the posture analysis:
def calibration_get_threshold(head_positions, distances):
    return np.mean(head_positions) + np.mean(distances)


# --- functions for posture analysis:

# inspect the posture by four specific criteria:
def check_posture(key_points, max_scores, face_center_threshold, confidence_threshold=0.1):
    posture_list = [False, False, False, False]
    xm, ym = get_mean_face_positions(key_points)

    # --- check if head twisted
    # if one ear is not visible:
    if max_scores[16] < confidence_threshold or max_scores[17] < confidence_threshold:
        posture_list[0] = True
    # if distance between ear and face center is too large (i.e. found ear at wrong location because it is invisible):
    if distance([ym, xm], key_points[16]) > 2 * distance(key_points[14], key_points[15]):
        posture_list[0] = True
    if distance([ym, xm], key_points[17]) > 2 * distance(key_points[14], key_points[15]):
        posture_list[0] = True

    # --- check if head leaned forward
    # if both ears y values are above the eyes y values:
    if key_points[16, 0] < key_points[14, 0] and key_points[17, 0] < key_points[15, 0]:
        posture_list[1] = True

    # --- check if head leaned sidewards
    # if the angle between the eyes is too high or too low:
    angle = 90 + 180 / np.pi * np.arctan2(key_points[15, 0] - key_points[14, 0], key_points[15, 1] - key_points[14, 1])
    if angle > 100 or angle < 80:
        posture_list[2] = True

    # --- check if the head is ducked
    # if the center of the face is below a threshold:
    if ym > face_center_threshold:
        posture_list[3] = True

    return posture_list


# --- functions for eye detection and eye status monitoring:

# use binary classifier to check whether the eyes are opened or closed:
def eye_status(img_l_eye, img_r_eye):

    # handle error appearing if the eye position is not detected:
    if 0 in img_l_eye.shape or 0 in img_r_eye.shape:
        return -1, -1

    img_l_eye = cv.resize(img_l_eye, (64, 64))
    img_r_eye = cv.resize(img_r_eye, (64, 64))

    img_l_eye = np.expand_dims(img_l_eye, axis=0)
    img_r_eye = np.expand_dims(img_r_eye, axis=0)

    tensor = np.append(img_l_eye, img_r_eye, axis=0)

    y = EYE_CLASSIFIER.apply(tensor)

    l_eye_open = y[0, 0]
    r_eye_open = y[1, 0]

    return l_eye_open, r_eye_open


# create and handle list of previous eye states:
def track_eye_status(l_status, r_status, l_new, r_new):
    l_status.append(l_new)
    r_status.append(r_new)
    if len(l_status) > 10:
        l_status.pop(0)
        r_status.pop(0)
    return l_status, r_status


# detect blinking by evaluating the status of an eye over time:
# a blinking is detected if a series of opened-closed-opened can be found with each segment lasting for a specific time
def check_blink(status):
    threshold = 0.5
    opened_before_counter = 0
    closed_counter = 0
    opened_after_counter = 0
    for t in status:
        if opened_before_counter >= 2:
            if closed_counter >= 2:
                if t <= threshold:
                    opened_after_counter += 1
                else:
                    if opened_after_counter == 0:
                        closed_counter += 1
                    else:
                        opened_before_counter = 0
                        closed_counter = 0
                        opened_after_counter = 0
                if closed_counter >= 6:
                    opened_before_counter = 0
                    closed_counter = 0
                    opened_after_counter = 0
            else:
                if t >= threshold:
                    closed_counter += 1
                else:
                    if closed_counter == 0:
                        opened_before_counter += 1
                    else:
                        closed_counter = 0
                        opened_before_counter = 1
        else:
            if t <= threshold:
                opened_before_counter += 1
            else:
                opened_before_counter = 0
                closed_counter = 0
                opened_after_counter = 0
        if opened_before_counter >= 2 and closed_counter >= 2 and opened_after_counter >= 2:
            return True
    return False


# evaluate whether the event of right-left-blink was detected:
def check_right_left_blink(l_status, r_status, timer):
    rl_blink = False

    # if right blink was already detected:
    if timer < 0:
        # if right blink was detected too long ago reset:
        if timer < -10:
            timer = 1

        # if right blink was not too recently check if left blink is detected:
        if timer < -3:
            if check_blink(l_status[:int(-timer)]):
                rl_blink = True
                timer = 0
                l_status = []
                r_status = []

        timer -= 1

    # if no right blink was detected yet, check for right blink:
    if timer == 0:
        if check_blink(r_status):
            timer = -1

    return rl_blink, timer, l_status, r_status






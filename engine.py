# calculations on the networks output

import numpy as np

import pose_net


POSE_NET = pose_net.build_net()


def mirror_vertical(image):
    return np.array(image[:, ::-1, :])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def distance(p1, p2):
    return np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)


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
# TODO: improve calculation of height h
# TODO: consider rotations of the head
def find_face(key_points):
    x, y = get_mean_face_positions(key_points)
    w = 0.5 * distance(key_points[16], key_points[17])
    h = w * 1.5
    angle = 90 + 180 / np.pi * np.arctan2(key_points[15, 0] - key_points[14, 0], key_points[15, 1] - key_points[14, 1])
    return x, y, h, w, angle


# control the posture by specific criteria
def check_posture(key_points, max_scores, confidence_threshold=0.1, face_center_threshold=260):
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



# calculations on the networks output

import numpy as np

import pose_net


POSE_NET = pose_net.build_net()


def mirror_vertical(image):
    return np.array(image[:, ::-1, :])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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
    w = 0.5 * np.sqrt((key_points[17, 0] - key_points[16, 0])**2 + (key_points[17, 1] - key_points[16, 1])**2)
    h = w * 1.5
    angle = 90 + 180 / np.pi * np.arctan2(key_points[15, 0] - key_points[14, 0], key_points[15, 1] - key_points[14, 1])
    return x, y, h, w, angle



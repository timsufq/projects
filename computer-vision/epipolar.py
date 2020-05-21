#!/usr/bin/env python3
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import svd

def fundamental_matrix(matches):
    # raise NotImplementedError
    # img1 img2: RGB image
    # matches: 2D numpy array, “ x1 y1 x2 y2 ”  matches_num * 4

    matches_num = matches.shape[0]
    list_of_matches = matches.tolist()
    l_points = np.concatenate((matches[:, 0:2], np.ones((matches_num, 1))), axis=1)
    r_points = np.concatenate((matches[:, 2:], np.ones((matches_num, 1))), axis=1)

    # RANSAC
    sample_num = 10
    matches_bool = [False] * matches_num
    inliers = None
    num_inlier = 0
    iteration = 50000
    threshold = 10

    for rnd in range(iteration):
        selected = random.sample(list_of_matches, sample_num)
        # print(selected)
        # compute fundamental matrix
        A = np.zeros((sample_num, 9))
        for i in range(sample_num):
            x_l, y_l, x_r, y_r = selected[i][0], selected[i][1], selected[i][2], selected[i][3]
            A[i, :] = np.array([x_l * x_r, x_l * y_r, x_l, x_r * y_l, y_l * y_r, y_l, x_r, y_r, 1])
        _, _, V1 = svd(A)
        F = V1[-1].reshape(3, 3)
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = np.dot(U, np.dot(np.diag(S), V))
        F = F / F[2, 2]

        F_1 = np.dot(l_points, F)
        F_2 = np.dot(F, r_points.T)
        denominator = F_1[:, 0] ** 2 + F_1[:, 1] ** 2 + F_2[0] ** 2 + F_2[1] ** 2
        e = (np.diag(np.dot(np.dot(l_points, F), r_points.T))) ** 2 / denominator
        # print(e)
        inlier_idx = np.where(e < threshold)[0]
        if len(inlier_idx) > num_inlier:
            # print(rnd)
            num_inlier = len(inlier_idx)
            inliers = inlier_idx

    # compute final F
    samples = []
    for k in inliers:
        samples.append(list_of_matches[k])
        matches_bool[k] = True
    A = np.zeros((num_inlier, 9))
    for i in range(num_inlier):
        x_l, y_l, x_r, y_r = samples[i][0], samples[i][1], samples[i][2], samples[i][3]
        tmp = [x_l * x_r, x_l * y_r, x_l, x_r * y_l, y_l * y_r, y_l, x_r, y_r, 1]
        A[i, :] = np.array(tmp)
    _, _, v = svd(A)
    F = v[-1].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F = F / F[2, 2]
    # print(F, matches_bool)
    return F, matches_bool

def draw_epipolar_lines(F, image1, image2, matches, matches_bool, feature_num):
    # raise NotImplementedError
    # matches_bool: boolean list
    M1, N1, D = image1.shape
    M2, N2, D = image2.shape
    matched_img = np.zeros((max(M1, M2), N1+N2, 3), dtype=np.uint8)
    matched_img[0:M1, 0:N1, :] = image1
    matched_img[0:M2, N1:, :] = image2
    
    inliers = []
    for i in range(len(matches)):
        if matches_bool[i] == True:
            inliers.append(matches[i])

    picked_feature = np.array(random.sample(inliers, feature_num))
    left_points = np.concatenate((picked_feature[:, 0:2], np.ones((feature_num, 1))), axis=1)
    right_points = np.concatenate((picked_feature[:, 2:], np.ones((feature_num, 1))), axis=1)
    right_lines = np.dot(F.T, left_points.T)
    for j in range(feature_num):
        x0,y0 = map(int, [0, -right_lines[2, j] / right_lines[1, j]])
        x1,y1 = map(int, [N2, -(right_lines[2, j] + right_lines[0, j] * N2) / right_lines[1, j]])
        matched_img = cv2.circle(matched_img, (int(left_points[j, 0]), int(left_points[j, 1])), 5, color=(0, 255, 0), thickness=3)
        matched_img = cv2.circle(matched_img, (int(right_points[j, 0] + N1), int(right_points[j, 1])), 5, color=(0, 255, 255), thickness=3)
        matched_img = cv2.line(matched_img, (x0 + N1,y0), (x1 + N1,y1), color=(0, 255, 0), thickness=2)
    return matched_img

def main():
    img_path1 = 'data/hopkins1.jpg'
    img_path2 = 'data/hopkins2.jpg'
    file_pth = 'data/matches.txt'
    feature_num = 8

    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    # read matches
    matches = []
    for line in open(file_pth, 'r'):
        data = line[:-2].split(' ')
        for i in range(len(data)):
            data[i] = float(data[i])
        matches.append(data)
    matches = np.array(matches)
    # compute fundamental matrix
    F, matches_bool = fundamental_matrix(matches)
    print(F)
    # epipolar lines
    epipolar_lines = draw_epipolar_lines(F, img1, img2, matches, matches_bool, feature_num)
    plt.figure(1)
    plt.imshow(cv2.cvtColor(epipolar_lines, cv2.COLOR_BGR2RGB))
    cv2.imwrite("epipolar_lines.png", epipolar_lines)

if __name__ == '__main__':
  main()

#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy import signal


def detect_corners(image):
  """Harris corner detector.

  Args:
  - image (2D float64 array): A grayscale image.

  Returns:
  - corners (list of 2-tuples): A list of 2-tuples representing the locations
      of detected corners. Each tuple contains the (x, y) coordinates of a
      pixel, where y is row index and x is the column index, i.e. `image[y, x]`
      gives the corresponding pixel intensity.
  """
  # SELF-DEFINED
  k = 0.05
  score_threshold = 0.01 # of the max R value in the image
  gaussian_filter_size = 3
  gaussian_filter_std = 1
  # SELF-DEFINED

  # Sobel Operator
  sobel_dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  sobel_dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

  # Gaussian Filter
  gaussian_filter = np.zeros([gaussian_filter_size, gaussian_filter_size])
  filter_midpoint = gaussian_filter_size // 2
  for x in range(gaussian_filter_size):
    for y in range(gaussian_filter_size):
      gaussian_filter[x][y] = np.exp(- ((x - filter_midpoint) ** 2 + (y - filter_midpoint) ** 2) / (2 * gaussian_filter_std ** 2))
  gaussian_filter /= np.sum(gaussian_filter)

  # Gray image is already got from main

  # Self-written convolution is correct but too much slower, therefore use scipy.signal.convolve2d
  ix = signal.convolve2d(image, sobel_dx, mode = 'same')
  iy = signal.convolve2d(image, sobel_dy, mode = 'same')
  A = signal.convolve2d(ix ** 2, gaussian_filter, mode = 'same')
  B = signal.convolve2d(iy ** 2, gaussian_filter, mode = 'same')
  C = signal.convolve2d(ix * iy, gaussian_filter, mode = 'same')

  # # Self-written convolution
  # ix = kernel_conv(image, sobel_dx)
  # iy = kernel_conv(image, sobel_dy)
  # A = kernel_conv(ix ** 2, gaussian_filter)
  # B = kernel_conv(iy ** 2, gaussian_filter)
  # C = kernel_conv(ix * iy, gaussian_filter)

  R = A * B - C ** 2 - k * (A + B) ** 2 # Harris and Stephens 1988

  R[R < score_threshold * np.max(R)] = 0

  corners = np.argwhere(nonmaxsup(R, 3) > 0)
  corners[:,[0, 1]] = corners[:,[1, 0]]
  corners = [tuple(corner) for corner in corners]

  return corners


# ! Not used. Very time consuming.
def kernel_conv(image, kernel):
  # http://www.songho.ca/dsp/convolution/convolution.html
  result = np.zeros(image.shape)

  kernel_center_x = kernel.shape[0] // 2
  kernel_center_y = kernel.shape[1] // 2

  for x in range(result.shape[0]):
    for y in range(result.shape[1]):
      for m in range(kernel.shape[0]):
        mm = kernel.shape[0] - 1 - m
        xx = x + kernel_center_x - mm
        if 0 <= xx < result.shape[0]:
          for n in range(kernel.shape[1]):
            nn = kernel.shape[1] - 1 - n
            yy = y + kernel_center_y - nn
            if 0 <= yy < result.shape[1]:
              result[x][y] += image[xx][yy] * kernel[mm][nn]

  return result


def nonmaxsup(scores, ksize):
  """Apply non-maximum suppression on a score map.

  Args:
  - scores (2D float64 array): A score map.
  - ksize (int): Kernel size of the maximum filter being used.

  Returns:
  - suppressed (2D float64 array): Suppressed version of `scores` where all
      elements except the local maxima are set to 0.
  """
  suppressed = np.copy(scores)
  filtered = maximum_filter(suppressed, (ksize, ksize))
  maxima = (suppressed == filtered)
  suppressed[np.logical_not(maxima)] = 0
  return suppressed


def match_corners(image1, image2, corners1, corners2):
  """Match corners using mutual marriages.

  Args:
  - image1 (2D float64 array): A grayscale image.
  - image2 (2D float64 array): A grayscale image.
  - corners1 (list of 2-tuples): Corners in image1.
  - corners2 (list of 2-tuples): Corners in image2.

  Returns:
  - matches (list of 2-tuples): A list of 2-tuples representing the matching
      indices. Each tuple contains two integer indices. For example, tuple
      (0, 42) indicates that corners1[0] is matched to corners2[42].
  """
  # SELF-DEFINED
  win_radius = 6 # window will be square with width of (win_radius * 2 + 1)
  corner_sample_limit = 500
  # SELF-DEFINED

  matches = []

  corners1 = [(i[1], i[0]) for i in corners1]
  corners2 = [(i[1], i[0]) for i in corners2]

  valid_corners1 = []
  valid_corners2 = []
  for i in corners1:
    if i[0] in range(win_radius, image1.shape[0] - 1 - win_radius) and i[1] in range(win_radius, image1.shape[1] - 1 - win_radius):
      valid_corners1.append(i)
  for i in corners2:
    if i[0] in range(win_radius, image2.shape[0] - 1 - win_radius) and i[1] in range(win_radius, image2.shape[1] - 1 - win_radius):
      valid_corners2.append(i)
  
  if len(valid_corners1) > corner_sample_limit:
	  valid_corners1 = [valid_corners1[i] for i in np.random.choice(len(valid_corners1), corner_sample_limit, replace = False)]
  if len(valid_corners2) > corner_sample_limit:
	  valid_corners2 = [valid_corners2[i] for i in np.random.choice(len(valid_corners2), corner_sample_limit, replace = False)]

  matches1 = ncc_computation(image1, image2, valid_corners1, valid_corners2, win_radius)
  matches2 = ncc_computation(image2, image1, valid_corners2, valid_corners1, win_radius)

  for i in matches1:
    for j in i[1]:
      j_matches = []
      for k in matches2:
        if k[0] == j:
          j_matches = k[1]
          break
      if i[0] in j_matches:
        matches.append((corners1.index(i[0]), corners2.index(j)))
        
  return matches


def ncc_computation(image1, image2, valid_corners1, valid_corners2, win_radius):
  matches = []

  for i in valid_corners1:
    best_match = []
    best_score = -1 # NCC score ranges in [-1, 1]

    i_win = image1[(i[0] - win_radius):(i[0] + win_radius), (i[1] - win_radius):(i[1] + win_radius)]
    i_win = (i_win - np.mean(i_win)) / np.std(i_win)

    for j in valid_corners2:
      j_win = image2[(j[0] - win_radius):(j[0] + win_radius), (j[1] - win_radius):(j[1] + win_radius)]
      j_win = (j_win - np.mean(j_win)) / np.std(j_win)

      score = np.sum(np.multiply(i_win, j_win)) / ((win_radius * 2 + 1) ** 2)

      if score > best_score:
        best_match = [j]
        best_score = score
      elif score == best_score:
        best_match.append(j)

    if best_match:
      matches.append([i, best_match])

  return matches


def draw_matches(image1, image2, corners1, corners2, matches,
    outlier_labels=None):
  """Draw matched corners between images.

  Args:
  - matches (list of 2-tuples)
  - image1 (3D uint8 array): A color image having shape (H1, W1, 3).
  - image2 (3D uint8 array): A color image having shape (H2, W2, 3).
  - corners1 (list of 2-tuples)
  - corners2 (list of 2-tuples)
  - outlier_labels (list of bool)

  Returns:
  - match_image (3D uint8 array): A color image having shape
      (max(H1, H2), W1 + W2, 3).
  """
  img1 = image1.copy()
  img2 = image2.copy()

  if img1.shape[0] > img2.shape[0]:
    img2 = np.pad(img2, ((0, (img1.shape[0] - img2.shape[0])), (0, 0), (0, 0)), mode = 'constant')
  elif img1.shape[0] < img2.shape[0]:
    img1 = np.pad(img1, ((0, (img2.shape[0] - img1.shape[0])), (0, 0), (0, 0)), mode = 'constant')

  match_image = cv2.hconcat([img1, img2])
  
  if outlier_labels:
    for i in range(len(matches)):
      cm1 = corners1[matches[i][0]] # coordinates (x, y) of cv2 using horizontal as x and vertical as y
      cm2 = (corners2[matches[i][1]][0] + img1.shape[1], corners2[matches[i][1]][1])

      cv2.circle(match_image, cm1, 5, (0, 255, 0), 1) # color: BGR
      cv2.circle(match_image, cm2, 5, (0, 255, 0), 1)

      if outlier_labels[i]:
        cv2.line(match_image, cm1, cm2, (0, 0, 255), 1)
      else:
        cv2.line(match_image, cm1, cm2, (255, 0, 0), 1)

  else:
    for match in matches:
      cm1 = corners1[match[0]] # coordinates (x, y) of cv2 using horizontal as x and vertical as y
      cm2 = (corners2[match[1]][0] + img1.shape[1], corners2[match[1]][1])

      cv2.circle(match_image, cm1, 5, (0, 255, 0), 1) # color: BGR
      cv2.circle(match_image, cm2, 5, (0, 255, 0), 1)
      cv2.line(match_image, cm1, cm2, (255, 0, 0), 1)
  
  return match_image


def compute_affine_xform(corners1, corners2, matches):
  """Compute affine transformation given matched feature locations.

  Args:
  - corners1 (list of 2-tuples)
  - corners2 (list of 2-tuples)
  - matches (list of 2-tuples)

  Returns:
  - xform (2D float64 array): A 3x3 matrix representing the affine
      transformation that maps coordinates in image1 to the corresponding
      coordinates in image2.
  - outlier_labels (list of bool): A list of Boolean values indicating whether
      the corresponding match in `matches` is an outlier or not. For example,
      if `matches[42]` is determined as an outlier match after RANSAC, then
      `outlier_labels[42]` should have value `True`.
  """
  # SELF-DEFINED
  inlier_threshold = 3 # shift tolerance of corners measured in pixels
  round_time = 50
  # SELF-DEFINED

  round_affine_matrix_sets = []
  round_outliers_sets = []
  round_inliers_number_sets = []

  for round_i in range(round_time):
    round_inliers = []
    round_outliers = []

    # 1. Randomly choose s samples (6 for affine transformation, which is 3 pairs)
    trans_match = [matches[i] for i in np.random.choice(len(matches), 3, replace = False)]
    # ? redo if repeated

    # 2. Fit a model to those samples
    A = np.array([
      [corners1[trans_match[0][0]][0], corners1[trans_match[0][0]][1], 1],
      [corners1[trans_match[1][0]][0], corners1[trans_match[1][0]][1], 1],
      [corners1[trans_match[2][0]][0], corners1[trans_match[2][0]][1], 1]
      ])
    B = np.array([[corners2[trans_match[0][1]][0]], [corners2[trans_match[1][1]][0]], [corners2[trans_match[2][1]][0]]])
    A_inv = np.linalg.inv(A)
    abc_vector = A_inv.dot(B).T

    F = np.array([[corners2[trans_match[0][1]][1]], [corners2[trans_match[1][1]][1]], [corners2[trans_match[2][1]][1]]])
    def_vector = A_inv.dot(F).T

    affine_matrix = np.vstack([abc_vector, def_vector, [0, 0, 1]]) # Affine matrix is coverting image1 to image2

    # 3. Count the number of inliers that approximately fit the model
    for i in range(len(matches)):
      weighted_c1 = np.append(np.array(list(corners1[matches[i][0]])), 1).T
      weighted_est_c2 = affine_matrix.dot(weighted_c1)
      est_c2 = np.delete(weighted_est_c2, 2)

      err = np.sqrt(np.sum((est_c2 - corners2[matches[i][1]]) ** 2))
      if err <= inlier_threshold:
        round_inliers.append(i)
      else:
        round_outliers.append(i)
    
    round_affine_matrix_sets.append(affine_matrix)
    round_outliers_sets.append(round_outliers)
    round_inliers_number_sets.append(len(round_inliers))
  
  # Choose the best model
  best_idx = np.argmax(round_inliers_number_sets)
  xform = round_affine_matrix_sets[best_idx]
  outlier_labels = np.zeros(len(matches), dtype = bool)
  np.put(outlier_labels, round_outliers_sets[best_idx], True)
  outlier_labels = outlier_labels.tolist()

  return xform, outlier_labels


# Extra Credit
def compute_proj_xform(corners1, corners2, matches):
  """Compute projective transformation given matched feature locations.

  Args:
  - corners1 (list of 2-tuples)
  - corners1 (list of 2-tuples)
  - matches (list of 2-tuples)

  Returns:
  - xform (2D float64 array): A 3x3 matrix representing the projective
      transformation that maps coordinates in image1 to the corresponding
      coordinates in image2.
  - outlier_labels (list of bool)
  """
  raise NotImplementedError


def stitch_images(image1, image2, xform):
  """Stitch two matched images given the transformation between them.

  Args:
  - image1 (3D uint8 array): A color image.
  - image2 (3D uint8 array): A color image.
  - xform (2D float64 array): A 3x3 matrix representing the transformation
      between image1 and image2. This transformation should map coordinates
      in image1 to the corresponding coordinates in image2.

  Returns:
  - image_stitched (3D uint8 array)
  """
  # SELF-DEFINED
  op_rate = 0.5
  saturation_rate = 1
  # SELF-DEFINED
  
  warped_image1 = cv2.warpAffine(image1, xform[0:2, :], (image2.shape[1], image2.shape[0]))
  image_stitched = np.zeros(image2.shape)

  for x in range(image2.shape[0]):
    for y in range(image2.shape[1]):
      # Image Saturation, sequence: BGR
      image_stitched[x][y][0] = warped_image1[x][y][0] * op_rate + image2[x][y][0] * (1 - op_rate)
      image_stitched[x][y][1] = warped_image1[x][y][1] * op_rate + image2[x][y][1] * (1 - op_rate)
      image_stitched[x][y][2] = warped_image1[x][y][2] * op_rate + image2[x][y][2] * (1 - op_rate)

      luminance = 0.11 * image_stitched[x][y][0] + 0.59 * image_stitched[x][y][1] + 0.30 * image_stitched[x][y][2]

      image_stitched[x][y][0] = (image_stitched[x][y][0] - luminance) * saturation_rate + luminance
      image_stitched[x][y][1] = (image_stitched[x][y][1] - luminance) * saturation_rate + luminance
      image_stitched[x][y][2] = (image_stitched[x][y][2] - luminance) * saturation_rate + luminance

  # Can be replaced with below statemnet
  # image_stitched = cv2.addWeighted(warped_image1, op_rate, image2, 1 - op_rate, 0)

  image_stitched = image_stitched.astype('uint8')

  return image_stitched


# Extra Credit
def detect_blobs(image):
  """Laplacian blob detector.

  Args:
  - image (2D float64 array): A grayscale image.

  Returns:
  - corners (list of 2-tuples): A list of 2-tuples representing the locations
      of detected blobs. Each tuple contains the (x, y) coordinates of a
      pixel, which can be indexed by image[y, x].
  - scales (list of floats): A list of floats representing the scales of
      detected blobs. Has the same length as `corners`.
  - orientations (list of floats): A list of floats representing the dominant
      orientation of the blobs.
  """
  raise NotImplementedError


# Extra Credit
def compute_descriptors(image, corners, scales, orientations):
  """Compute descriptors for corners at specified scales.

  Args:
  - image (2d float64 array): A grayscale image.
  - corners (list of 2-tuples): A list of (x, y) coordinates.
  - scales (list of floats): A list of scales corresponding to the corners.
      Must have the same length as `corners`.
  - orientations (list of floats): A list of floats representing the dominant
      orientation of the blobs.

  Returns:
  - descriptors (list of 1d array): A list of desciptors for each corner.
      Each element is an 1d array of length 128.
  """
  if len(corners) != len(scales) or len(corners) != len(orientations):
    raise ValueError(
        '`corners`, `scales` and `orientations` must all have the same length.')

  raise NotImplementedError


# Extra Credit
def match_descriptors(descriptors1, descriptors2):
  """Match descriptors based on their L2-distance and the "ratio test".

  Args:
  - descriptors1 (list of 1d arrays):
  - descriptors2 (list of 1d arrays):

  Returns:
  - matches (list of 2-tuples): A list of 2-tuples representing the matching
      indices. Each tuple contains two integer indices. For example, tuple
      (0, 42) indicates that corners1[0] is matched to corners2[42].
  """
  raise NotImplementedError


def baseline_main(argv):
  data_path = 'data/'

  img_path1 = data_path + argv[0] + '.png'
  img_path2 = data_path + argv[1] + '.png'

  img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
  img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0

  # TODO

  gray1 = np.float32(gray1)
  gray2 = np.float32(gray2)

  # Feature Detection
  corners1 = detect_corners(gray1)
  corners2 = detect_corners(gray2)

  # Feature Matching
  matches = match_corners(gray1, gray2, corners1, corners2)

  # Image Alignment
  xform, outlier_labels = compute_affine_xform(corners1, corners2, matches)

  # Displaying Matches
  match_image = draw_matches(img1, img2, corners1, corners2, matches, outlier_labels)
  cv2.imwrite('output_matched_image_' + argv[0] + '_' + argv[1] + '.png', match_image)

  # Image Stitching
  image_stitched = stitch_images(img1, img2, xform)
  cv2.imwrite('output_stitched_image_' + argv[0] + '_' + argv[1] + '.png', image_stitched)


# Extra Credit
def extra_main():
  pass


if __name__ == '__main__':
  baseline_main(sys.argv[1:])
  # extra_main()


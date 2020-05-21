#!/usr/bin/env python3
import cv2
import numpy as np
import sys

def detect_edges(image):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
  - edge_image (2D float array): A heat map where the intensity at each point
      is proportional to the edge magnitude.
  """
  edge_image = np.zeros(image.shape)
  image = np.vstack((image, np.zeros(image.shape[1])))
  image = np.concatenate((image, np.zeros((image.shape[0], 1))), axis = 1)

  sobel_dx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
  sobel_dy = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

  for x in range(edge_image.shape[0]):
    for y in range(edge_image.shape[1]):
      x_direction = \
        sobel_dx[0][0] * image[x - 1][y - 1] + \
        sobel_dx[0][1] * image[x    ][y - 1] + \
        sobel_dx[0][2] * image[x + 1][y - 1] + \
        sobel_dx[1][0] * image[x - 1][y    ] + \
        sobel_dx[1][1] * image[x    ][y    ] + \
        sobel_dx[1][2] * image[x + 1][y    ] + \
        sobel_dx[2][0] * image[x - 1][y + 1] + \
        sobel_dx[2][1] * image[x    ][y + 1] + \
        sobel_dx[2][2] * image[x + 1][y + 1]
      y_direction = \
        sobel_dy[0][0] * image[x - 1][y - 1] + \
        sobel_dy[0][1] * image[x    ][y - 1] + \
        sobel_dy[0][2] * image[x + 1][y - 1] + \
        sobel_dy[1][0] * image[x - 1][y    ] + \
        sobel_dy[1][1] * image[x    ][y    ] + \
        sobel_dy[1][2] * image[x + 1][y    ] + \
        sobel_dy[2][0] * image[x - 1][y + 1] + \
        sobel_dy[2][1] * image[x    ][y + 1] + \
        sobel_dy[2][2] * image[x + 1][y + 1]
      edge_image[x][y] = np.sqrt(x_direction **2 + y_direction ** 2)

  return edge_image

def hough_circles(edge_image, edge_thresh, radius_values):
  """Threshold edge image and calculate the Hough transform accumulator array.

  Args:
  - edge_image (2D float array): An H x W heat map where the intensity at each
      point is proportional to the edge magnitude.
  - edge_thresh (float): A threshold on the edge magnitude values.
  - radius_values (1D int array): An array of R possible radius values.

  Return:
  - thresh_edge_image (2D bool array): Thresholded edge image indicating
      whether each pixel is an edge point or not.
  - accum_array (3D int array): Hough transform accumulator array. Should have
      shape R x H x W.
  """
  thresh_edge_image = edge_image >= edge_thresh
  accum_array = np.zeros((len(radius_values), thresh_edge_image.shape[0], thresh_edge_image.shape[1]), dtype = int)
  theta = np.arange(0, 2 * np.pi, 2 * np.pi / 100)

  detector = []
  for i in range(len(radius_values)):
    for t in theta:
      detector.append([i, radius_values[i] * np.cos(t), radius_values[i] * np.sin(t)])

  for x in range(thresh_edge_image.shape[0]):
    for y in range(thresh_edge_image.shape[1]):
      if thresh_edge_image[x][y]:
        for d in detector:
          dx = int(x + d[1])
          dy = int(y + d[2])
          if (0 <= dx < thresh_edge_image.shape[0]) and \
            (0 <= dy < thresh_edge_image.shape[1]):
            accum_array[d[0]][dx][dy] += 1

  return thresh_edge_image, accum_array

def find_circles(image, accum_array, radius_values, hough_thresh):
  """Find circles in an image using output from Hough transform.

  Args:
  - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
      original color image instead of its grayscale version so the circles
      can be drawn in color.
  - accum_array (3D int array): Hough transform accumulator array having shape
      R x H x W.
  - radius_values (1D int array): An array of R radius values.
  - hough_thresh (int): A threshold of votes in the accumulator array.

  Return:
  - circles (list of 3-tuples): A list of circle parameters. Each element
      (r, y, x) represents the radius and the center coordinates of a circle
      found by the program.
  - circle_image (3D uint8 array): A copy of the original image with detected
      circles drawn in color.
  """
  circles = []
  circle_image = image
  
  for r in range(accum_array.shape[0]):
    for x in range(accum_array.shape[1]):
      for y in range(accum_array.shape[2]):
        if accum_array[r][x][y] > hough_thresh:
          circles.append((radius_values[r], y, x))

  for i in range(len(circles)):
      cv2.circle(circle_image, (circles[i][1], circles[i][2]), circles[i][0], (0, 255, 0), thickness = 2)

  return circles, circle_image

def main(argv):
  img_name = argv[0]

  # 1 - edge_ image generation
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edge_image = detect_edges(gray_image)

  # 2 - accum_array generation
  max_g = int(np.max(edge_image))
  edge_thresh = max_g * 0.5
  radius_values = np.arange(20, 41)
  thresh_edge_image, accum_array = hough_circles(edge_image, edge_thresh, radius_values)

  # 3 - circle_image generation
  hough_thresh = np.max(accum_array) * 0.5
  circles, circle_image = find_circles(img, accum_array, radius_values, hough_thresh)
  # 

  # 4 - print results
  print(circles)
  for x in range(edge_image.shape[0]):
    for y in range(edge_image.shape[1]):
      edge_image[x][y] = int(edge_image[x][y] / max_g * 255)
  cv2.imwrite('output/' + img_name + "_sobel.png", edge_image)
  cv2.imwrite('output/' + img_name + "_edges.png", thresh_edge_image * 255)
  cv2.imwrite('output/' + img_name + '_circles.png', circle_image)

if __name__ == '__main__':
  main(sys.argv[1:])

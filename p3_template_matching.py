#!/usr/bin/env python3
import cv2
import numpy as np
import sys

def normxcorr2(template, image):
  """Do normalized cross-correlation on grayscale images.

  When dealing with image boundaries, the "valid" style is used. Calculation
  is performed only at locations where the template is fully inside the search
  image.

  Args:
  - template (2D float array): Grayscale template image.
  - image (2D float array): Grayscale search image.

  Return:
  - scores (2D float array): Heat map of matching scores.
  """
  scores = np.zeros((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1))

  center_x = int(template.shape[0] / 2)
  center_y = int(template.shape[1] / 2)

  for x in range(scores.shape[0]):
    for y in range(scores.shape[1]):
      region = image[
          x : x + 2 * center_x + 1,
          y : y + 2 * center_y + 1,
          ]
      score = region.ravel() @ template.ravel().T
      nfactor = np.linalg.norm(region) * np.linalg.norm(template) + 0.00001
      scores[x][y] = score / nfactor

  return scores

def find_matches(template, image, thresh = None):
  """Find template in image using normalized cross-correlation.

  Args:
  - template (3D uint8 array): BGR template image.
  - image (3D uint8 array): BGR search image.

  Return:
  - coords (2-tuple or list of 2-tuples): When `thresh` is None, find the best
      match and return the (x, y) coordinates of the upper left corner of the
      matched region in the original image. When `thresh` is given (and valid),
      find all matches above the threshold and return a list of (x, y)
      coordinates.
  - match_image (3D uint8 array): A copy of the original search images where
      all matched regions are marked.
  """
  coords = []
  match_image = image
  
  nc_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  nc_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  nc_image = np.divide(nc_image, np.max(nc_image))
  nc_template = np.divide(nc_template, np.max(nc_template))
  scores = normxcorr2(nc_template, nc_image)

  if thresh == None:
    y, x = np.where(scores == np.max(scores))
    coords.append((int(x), int(y)))
  elif 0 < thresh <= 1:
    y, x = np.where(scores >= thresh)
    for i in range(len(x)):
      coords.append((int(x[i]), int(y[i])))

  for point in coords:
    cv2.rectangle(match_image, (point[0], point[1]), (point[0] + template.shape[1], point[1] + template.shape[0]), color=(0, 255, 0), thickness = 2)
  
  if len(coords) == 1:
    coords = coords[0]

  return coords, match_image

def main(argv):
  img_name = argv[0]
  tpl_name = argv[1]

  image = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  template = cv2.imread('data/' + tpl_name + '.png', cv2.IMREAD_COLOR)

  nc_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  nc_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  nc_image = np.divide(nc_image, np.max(nc_image))
  nc_template = np.divide(nc_template, np.max(nc_template))
  scores = normxcorr2(nc_template, nc_image)

  thresh_val = 0.98
  coords, match_image = find_matches(template, image, thresh = thresh_val)

  print(coords)
  heat_map = np.zeros(scores.shape)
  for x in range(scores.shape[0]):
    for y in range(scores.shape[1]):
      heat_map[x][y] = int(scores[x][y] * 255)
  cv2.imwrite('output/' + img_name + "_heat.png", heat_map)
  cv2.imwrite('output/' + img_name + ".png", match_image)

if __name__ == '__main__':
  main(sys.argv[1:])

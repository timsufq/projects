#!/usr/bin/env python3
import cv2
import numpy as np
import sys

def binarize(gray_image, thresh_val):
  return (gray_image >= thresh_val) * 255

def label(binary_image):
  current_label = 0
  labeled_image = np.full(binary_image.shape, 0, int)
  equivalent_records = np.empty((0, 2), int)

  # A - the first pass
  # 1 - the first row
  # 1.1 - the first point in the first row
  if binary_image[0][0] != 0:
    current_label += 1
    labeled_image[0][0] = current_label

  # 1.2 - all points in the first row except the first
  for k in range(1, binary_image.shape[1]):
    if binary_image[0][k] != 0:

      if labeled_image[0][k - 1] == 0:
        current_label += 1
        labeled_image[0][k] = current_label
      elif labeled_image[0][k - 1] != 0:
        labeled_image[0][k] = labeled_image[0][k - 1]

  # 2 - all rows except the first
  for x in range(1, binary_image.shape[0]):
    # 2.1 - the first column
    if binary_image[x][0] != 0:

      if labeled_image[x - 1][0] == 0:
        if labeled_image[x - 1][1] == 0:
          current_label += 1
          labeled_image[x][0] = current_label
        else:
          labeled_image[x][0] = labeled_image[x - 1][1]
      else:
        labeled_image[x][0] = labeled_image[x - 1][0]
          
    # 2.2 - all columns except the first and the last
    for y in range(1, binary_image.shape[1] - 1):
      if binary_image[x][y] != 0:

        if labeled_image[x][y - 1] == 0:
          if labeled_image[x - 1][y - 1] == 0:
            if labeled_image[x - 1][y] == 0:
              if labeled_image[x - 1][y + 1] == 0:
                current_label += 1
                labeled_image[x][y] = current_label
              else:
                labeled_image[x][y] = labeled_image[x - 1][y + 1]
            else:
              labeled_image[x][y] = labeled_image[x - 1][y]
          else:
            labeled_image[x][y] = labeled_image[x - 1][y - 1]
            if (labeled_image[x - 1][y + 1] != 0) and (labeled_image[x - 1][y + 1] != labeled_image[x - 1][y - 1]):
              equivalent_records = np.vstack((equivalent_records, [labeled_image[x - 1][y + 1], labeled_image[x - 1][y - 1]]))
        else:
          labeled_image[x][y] = labeled_image[x][y - 1]
          if (labeled_image[x - 1][y + 1] != 0) and (labeled_image[x - 1][y + 1] != labeled_image[x][y - 1]):
            equivalent_records = np.vstack((equivalent_records, [labeled_image[x - 1][y + 1], labeled_image[x][y - 1]]))

    # 2.3 - the last column
    if binary_image[x][binary_image.shape[1] - 1] != 0:

      if binary_image[x][binary_image.shape[1] - 2] == 0:
        if binary_image[x - 1][binary_image.shape[1] - 2] == 0:
          if binary_image[x - 1][binary_image.shape[1] - 1] == 0:
            current_label += 1
            labeled_image[x][binary_image.shape[1] - 1] = current_label
          else:
            labeled_image[x][binary_image.shape[1] - 1] = labeled_image[x - 1][binary_image.shape[1] - 1]
        else:
          labeled_image[x][binary_image.shape[1] - 1] = labeled_image[x - 1][binary_image.shape[1] - 2]
      else:
        labeled_image[x][binary_image.shape[1] - 1] = labeled_image[x][binary_image.shape[1] - 2]

  # B - the second pass
  # 3 - Deal with the equivalent table
  eq_groups = []
  for eq_relationship in equivalent_records:
    group_of_0 = -1
    group_of_1 = -1

    for i in range(len(eq_groups)):
      if eq_relationship[0] in eq_groups[i]:
        group_of_0 = i
        break

    for i in range(len(eq_groups)):
      if eq_relationship[1] in eq_groups[i]:
        group_of_1 = i
        break

    if (group_of_0 == -1) and (group_of_1 == -1):
      eq_groups.append([eq_relationship[0], eq_relationship[1]])
    elif (group_of_0 != -1) and (group_of_1 != -1):
      if group_of_0 != group_of_1:
        eq_groups[group_of_0] += eq_groups[group_of_1]
        eq_groups.remove(eq_groups[group_of_1])
    elif (group_of_0 != -1) and (group_of_1 == -1):
      eq_groups[group_of_0].append(eq_relationship[1])
    else:
      eq_groups[group_of_1].append(eq_relationship[0])

  equivalent_table = np.arange(current_label + 1, dtype = int)
  for group in eq_groups:
    min_lable = min(group)
    for label in group:
      equivalent_table[label] = min_lable

  table_simplification = np.sort(np.unique(equivalent_table))
  for i in range(equivalent_table.shape[0]):
    equivalent_table[i] = np.where(table_simplification == equivalent_table[i])[0]

  # 4 - Redraw the labeled_image with the equivalent values
  color_step = 255 / (table_simplification.shape[0] - 1)
  for x in range(labeled_image.shape[0]):
    for y in range(labeled_image.shape[1]):
      labeled_image[x][y] = equivalent_table[labeled_image[x][y]] * color_step

  return labeled_image

def get_attribute(labeled_image):
  attribute_list = []
  label_list = np.delete(np.unique(labeled_image), 0)
  object_list = []

  for i in range(len(label_list)):
    object_list.append([])
    for x in range(labeled_image.shape[0]):
      for y in range(labeled_image.shape[1]):
        if labeled_image[x][y] == label_list[i]:
          # convert (x, y) into (y, labeled_image.shape[1] - 1 - x)
          object_list[i].append([y, labeled_image.shape[1] - 1 - x])

  for i in range(len(object_list)):
    obj_dict = {'label':str(label_list[i])}

    # position
    area = len(object_list[i])
    x_p = y_p = 0
    for point in object_list[i]:
      x_p += point[0]
      y_p += point[1]
    x_p /= area
    y_p /= area
    
    obj_dict['position'] = {'x':x_p, 'y':y_p}

    # orientation
    a = b = c = 0
    for point in object_list[i]:
      a += (point[1] - y_p) ** 2
      b += 2 * (point[1] - y_p) * (point[0] - x_p)
      c += (point[0] - x_p) ** 2
    theta_1 = np.arctan2(b, a - c) / 2
    theta_2 = theta_1 + np.pi/2
    obj_dict['orientation'] = float(theta_1)

    # roundedness
    E_1 = ((a * np.sin(theta_1) ** 2) - (b * np.sin(theta_1) * np.cos(theta_1)) + (c * np.cos(theta_1) ** 2))
    E_2 = ((a * np.sin(theta_2) ** 2) - (b * np.sin(theta_2) * np.cos(theta_2)) + (c * np.cos(theta_2) ** 2))
    obj_dict['roundedness'] = float(E_1 / E_2)

    attribute_list.append(obj_dict)

  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])

  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  print(attribute_list)

if __name__ == '__main__':
  main(sys.argv[1:])

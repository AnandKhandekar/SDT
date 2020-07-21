import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt
from trp_helper import *
import sys

original_image_BGR = cv2.imread('calib2.jpg')
original_image_RGB = cv2.cvtColor(original_image_BGR, cv2.COLOR_BGR2RGB)
main_header = cv2.imread('main_header.jpg')

image_width = original_image_RGB.shape[1]
image_height = original_image_RGB.shape[0]

original_image_BGR_copy = original_image_BGR.copy()
original_image_RGB_copy = original_image_RGB.copy()

print('image Shape', original_image_RGB.shape)

# CHOOSE POINTS FOR Perspective
source_points = np.float32([[142., 298.],
                           [784., 315.],
                           [811., 371.],
                           [ 82., 347.]])
'''
source_points = np.float32([[600., 100.],
                           [1000., 100.],
                           [800., 300.],
                           [300.,300.]])  # shape of source_points is (4,2)
'''

for point in source_points:
    cv2.circle(original_image_RGB_copy, tuple(point), 8, (255, 255, 255), -1) # plot all 4 points

points = source_points.reshape((-1,1,2)).astype(np.int32) # reshape the source_points to (4,1,2)
cv2.polylines(original_image_RGB_copy, [points], True, (0,255,0), thickness=4) # draw the polygon

plt.figure(figsize=(12, 12))
plt.imshow(original_image_RGB_copy)
plt.show()

# DEFINE THE DESTINATION Points

src = source_points

dst=np.float32([(0.2,0.82), (0.80, 0.82), (0.80,0.87), (0.2,0.87)])
dst_size=(800,1080)
dst = dst * np.float32(dst_size)

H_matrix = cv2.getPerspectiveTransform(src, dst)
print("The perspective transform matrix:")
print(H_matrix)

# visualize the WARPED Perspective

warped = cv2.warpPerspective(original_image_RGB_copy, H_matrix, dst_size)

plt.figure(figsize=(12, 12))
plt.imshow(warped)
plt.show()

# CREATE YOLO MODEL AND SET parameters

confidence_threshold = 0.4
nms_threshold = 0.3

min_distance = 80
width = 608
height = 608

config = 'yolov3.cfg'
weights = 'yolov3.weights'
classes = 'coco.names'

with open(classes, 'rt') as f:  # open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None
    coco_classes = f.read().strip('\n').split('\n')

model = create_model(config, weights)
output_layers = get_output_layers(model)

#prediction with YOLO
blob = blob_from_image(original_image_RGB, (width, height))
outputs = predict(blob, model, output_layers)

# Get Detected Person Boxes
boxes = get_image_boxes(outputs, image_width, image_height, coco_classes)
print(len(boxes))

#Get Points as Birds Eye View
birds_eye_points = compute_point_perspective_transformation(H_matrix, boxes)
#print(len(birds_eye_points))

#Get Red and Green Box Cordinates¶
green_box, red_box = get_red_green_boxes(min_distance, birds_eye_points, boxes)


#Generate Birds-Eye-View Image
birds_eye_view_image = get_birds_eye_view_image(green_box, red_box,eye_view_height=image_height,eye_view_width=image_width//2)
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(birds_eye_view_image, cv2.COLOR_RGB2BGR))
plt.show()

#Draw red and green boxes on detected Human
box_red_green_image = get_red_green_box_image(original_image_BGR.copy(),green_box,red_box)
plt.figure(figsize=(20, 20))
plt.imshow(cv2.cvtColor(box_red_green_image, cv2.COLOR_RGB2BGR))
plt.show()

# Social Distance On Video


video = cv2.VideoCapture('street.mp4')
fps = video.get(cv2.CAP_PROP_FPS)  # calculates the  FPS of the input Video
writer = None
frame_number = 0
print('%-20s%-26s%-26s%-26s%-26s' % ('Processing Frame','| Total Detected Person','| Red Markerd Person','| Green Marked Person', '| Time per frame'))
while True:

  ret,frame = video.read()

  if not ret:
    break

  image_height, image_width = frame.shape[:2]

  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  blob = blob_from_image(image, (width, height))
  outputs = predict(blob, model, output_layers)
  boxes = get_image_boxes(outputs, image_width, image_height, coco_classes)
  birds_eye_points = compute_point_perspective_transformation(H_matrix, boxes)
  green_box, red_box = get_red_green_boxes(min_distance, birds_eye_points,boxes)
  birds_eye_view_image = get_birds_eye_view_image(green_box, red_box, eye_view_height=image_height,eye_view_width=image_width//2)
  box_red_green_image = get_red_green_box_image(frame.copy(), green_box,red_box)

  combined_image = np.concatenate((birds_eye_view_image,box_red_green_image), axis=1)
  main_header = cv2.resize(main_header,(combined_image.shape[1],main_header.shape[0]))
  dashboard_image  = np.concatenate((main_header,combined_image), axis=0)



  frame_number += 1
  time_per_frame = frame_number/fps
  sys.stdout.write('%-20i|%-25i|%-25i|%-25i|%-25i\n' % (frame_number,len(boxes),len(red_box),len(green_box),time_per_frame))

  if writer is None:
     fourcc = cv2.VideoWriter_fourcc(*"DIVX")
     writer = cv2.VideoWriter('result_street_v3.avi', fourcc, 15, (dashboard_image.shape[1], dashboard_image.shape[0]), True)

  writer.write(dashboard_image)

  del image,outputs,combined_image,dashboard_image,birds_eye_view_image

print(' ')
writer.release()
video.release()

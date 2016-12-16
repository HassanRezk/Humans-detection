from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

# Initialize People descriptor from cv2.
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the image and resize it to reduce detection time and improve detection accuracy.
image = cv2.imread("Test image.jpeg")
cv2.imshow("original image.", image)
image = imutils.resize(image, width=min(400, image.shape[1]))
original = image.copy()
 
# Detect all persons in the image.
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

# Draw the original bounding boxes.
for (x, y, w, h) in rects:
	cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
# Apply non-maxima suppression to the bounding boxes using a fairly large overlap threshold to try to maintain 
# overlapping boxes that are still people.
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs = None, overlapThresh = 0.65)

# Draw rectangles that surround each person.
for (xA, yA, xB, yB) in pick:
	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

# show the output image
cv2.imshow("After NMS", image)
cv2.waitKey(0)

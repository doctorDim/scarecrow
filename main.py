# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description='Persons detector.')
ap.add_argument("-i", "--images", required=False, help="path to images directory")
ap.add_argument("-v", "--videos", type = int, required=False, help="number of Web-camera")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector for images
def image_detect(path):
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# loop over the image paths
	for imagePath in paths.list_images(path):
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=min(800, image.shape[1]))
		orig = image.copy()

		# detect people in the image
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)

		# draw the original bounding boxes
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

		# draw the final bounding boxes
		for (xA, yA, xB, yB) in pick:
			cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
			# Center of rectangle
			centerX, centerY = int((xB - xA)/2+xA), int((yB - yA)/2+yA)
			cv2.circle(image, (centerX, centerY), 5, (0, 0, 255), 2)

		# show some information on the number of bounding boxes
		filename = imagePath[imagePath.rfind("/") + 1:]
		print("[INFO] {}: {} original boxes, {} after suppression".format(
			filename, len(rects), len(pick)))

		# show the output images
		#cv2.imshow("Before NMS", orig)
		cv2.imshow("After NMS", image)
		cv2.waitKey(0)


# initialize the HOG descriptor/person detector for video
def video(img):
	image = np.copy(img)
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
		# Center of rectangle
		centerX, centerY = int((xB - xA)/2+xA), int((yB - yA)/2+yA)
		cv2.circle(image, (centerX, centerY), 5, (0, 0, 255), 2)

	return image, centerX, centerY

if __name__ == '__main__':

	if args["images"] is not None:
		image_detect(args["images"])

	if args["videos"] is not None:
		cap = cv2.VideoCapture(args["videos"])

		while(cap.isOpened()):
			_, frame = cap.read()

			img = np.copy(frame)
			image, X, Y = video(img)
			print(X, Y)

			cv2.imshow('Video', image)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

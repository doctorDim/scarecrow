# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import serial

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

# Rotate camera
#def rotate():

if __name__ == '__main__':

	if args["images"] is not None:
		image_detect(args["images"])

	if args["videos"] is not None:
		cap = cv2.VideoCapture(args["videos"])

		# What is fps?
		#fps = cap.get(cv2.CAP_PROP_FPS)
		#print(fps)

		# Set new fps
		#cap.set(cv2.CAP_PROP_FPS, 5)
		#fps = cap.get(cv2.CAP_PROP_FPS)
		#print(fps)

		'''
	    # COM port settings
	    ser = serial.Serial()
	    ser.port = "/dev/ttyACM0"
	    ser.baudrate = 9600
	    ser.bytesize = serial.EIGHTBITS     #number of bits per bytes
	    ser.parity = serial.PARITY_NONE     #set parity check: no parity
	    ser.stopbits = serial.STOPBITS_ONE  #number of stop bits
	    #ser.timeout = None                 #block read
	    ser.timeout = 1                     #non-block read
	    #ser.timeout = 2                    #timeout block read
	    ser.xonxoff = False                 #disable software flow control
	    ser.rtscts = False                  #disable hardware (RTS/CTS) flow control
	    ser.dsrdtr = False                  #disable hardware (DSR/DTR) flow control
	    ser.writeTimeout = 2                #timeout for write
	    ser = serial.Serial('/dev/ttyACM0', 9600)
	    print('Enter 1 or 0...')
	    ser.write("1".encode())
	    '''

		# COM port
		#ser = serial.Serial('/dev/ttyACM0',9600,timeout=5)
	    #ser.isOpen()

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

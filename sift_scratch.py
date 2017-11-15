import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from scipy import signal

def sift():
	img = cv2.imread('7.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	plt.imshow(img, cmap='gray')
	plt.show()

	height, width = img.shape[:2]

	s = 3
	s1 = list(range(s+2))
	print(s1)

	sigma = 2.0

	octave = []
	nrows = 0
	# Gaussian pyramid
	while(height >=10 and width >= 10):
		scale = [2**(i/s)*sigma for i in s1]
		print(scale)
		print("scale length = {0}".format(len(scale)))
		#scaledGauss = [cv2.GaussianBlur(img, (5,5), sigmaX=i, sigmaY=i) for i in scale]
		kernel = [cv2.getGaussianKernel(5, i) for i in scale]
		scaledGauss = [signal.convolve2d(img, i, boundary='symm', mode='same') for i in kernel]
		print("Scaled Gauss = {0}".format(len(scaledGauss)))
		scaledGauss = [img] + scaledGauss
		print(len(scaledGauss))
		scaledGauss1 = scaledGauss[:-1]
		print(len(scaledGauss1))
		scaledGauss2 = scaledGauss[1:]
		print(len(scaledGauss2))
		DOG = [scaledGauss1[i] - scaledGauss2[i] for i in s1]
		print(DOG)
		octave.append(DOG)
		img = cv2.resize(scaledGauss[s],None, fx=0.5, fy=0.5)

		height, width = img.shape[:2]
		print(height,width)
		sigma = 2*sigma

		fig, axes = plt.subplots(1, s+3)

		for i in range(len(scaledGauss)):
			axes[i].imshow(scaledGauss[i], cmap='gray');
		
		fig2, axes2 = plt.subplots(1, s+2)

		for i in range(len(DOG)):
			axes2[i].imshow(DOG[i], cmap='gray');
		
		plt.show()

	# Keypoint Location
	for i in range(len(octave)):



if __name__ == '__main__':
	sift()

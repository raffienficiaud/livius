"""Tests for determining the appropriate color correction"""

import numpy as np
import os 
import cv2

import matplotlib.pyplot as plt

os.chdir('histograms')

# Setup arrays
hist_diffs_blue = np.empty(59)
hist_diffs_green = np.empty(59)
hist_diffs_red = np.empty(59)

hist_blues = []
hist_greens = []
hist_reds = []

for i in range(0,60):	
	# Load Histograms  
	hist_blues.append(np.load('histogram_blue' + str(i) + '.npy')) 
	hist_greens.append(np.load('histogram_green' + str(i) + '.npy'))
	hist_reds.append(np.load('histogram_red' + str(i) + '.npy'))

	# Calculate distance
	if i > 0:
		hist_diffs_blue[i-1] = cv2.compareHist(hist_blues[i-1], hist_blues[i], cv2.cv.CV_COMP_CHISQR)
		hist_diffs_green[i-1] = cv2.compareHist(hist_greens[i-1], hist_greens[i], cv2.cv.CV_COMP_CHISQR)
		hist_diffs_red[i-1] = cv2.compareHist(hist_reds[i-1], hist_reds[i], cv2.cv.CV_COMP_CHISQR)


# Output
print hist_diffs_blue
print hist_diffs_green
print hist_diffs_red

plt.plot(hist_diffs_blue, color='b')
plt.plot(hist_diffs_green, color='g')
plt.plot(hist_diffs_red, color='r')
plt.show()

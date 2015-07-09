from multiprocessing import Pool
import glob
import cv2
import re
import functools

import numpy as np
import matplotlib.pyplot as plt


def extract_info(file_name, **kwargs):
    t = get_time_from_filename(file_name)

    im = cv2.imread(file_name)
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # # color diff
    # im_diff = (im_lab - im0_lab) ** 2
    # im_diff_lab = np.sqrt(np.sum(im_diff, axis=2))

    # background
    # fgmask = self.fgbg.apply(im0)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.kernel)

    # threshold the diff
    # histogram
    # hist = []
    # for i in range(im_diff.shape[2]):
    #     hist.append(cv2.calcHist([im_diff], [i], None, [256], [0, 256]))

    hist_plane = []
    slide_hist_plane = []

    # Compute the histogram for the slide image
    resized_x = im.shape[1]
    resized_y = im.shape[0]

    slide_crop_coordinates = extraction_args['slide_coordinates']

    min_y = slide_crop_coordinates[0] * resized_y
    max_y = slide_crop_coordinates[1] * resized_y
    min_x = slide_crop_coordinates[2] * resized_x
    max_x = slide_crop_coordinates[3] * resized_x
    slide = im_gray[min_y : max_y, min_x : max_x]
    slidehist = cv2.calcHist([slide], [0], None, [256], [0, 256])

    # if t < 2:
    #     plt.subplot(2,1,1)
    #     plt.imshow(slide, cmap=plt.cm.Greys_r)
    #     plt.subplot(2,1,2)
    #     plt.plot(slidehist)
    #     plt.xlim([0,256])

    #     plt.show()

    # @todo(Stephan): Move this somewhere else?
    def get_histogram_min_max_boundaries_normalized(hist):
        """Gets the 1- and 99-percentile as an approximation of the boundaries
           of the histogram.

           Note:
                The Histogram is expected to be normalized

           Returns both the min and the max value for the histogram
        """
        t_min = 0
        t_max = 255

        min_mass = 0
        max_mass = 0

        # Integrate until we reach 1% of the mass from each direction
        while min_mass < 0.01:
            min_mass += hist[t_min]
            t_min += 1

        while max_mass < 0.01:
            max_mass += hist[t_max]
            t_max -= 1

        return t_min, t_max

    histogram_boundaries = get_histogram_min_max_boundaries_normalized(cv2.normalize(slidehist))

    return t, histogram_boundaries
    # return t, im_lab, histogram_boundaries


def _inner_rectangle(coordinates):
    """Get the inner rectangle of the slide coordinates for cropping the image.

       Returns a 4x1 numpy array in the order:
       [min_y, max_y, min_x, max_x]
    """

    # This is specified by the rectify_coordinates() function in slideDetection.py
    top_left = 0
    top_right = 1
    bottom_right = 2
    bottom_left = 3

    x = 0
    y = 1

    min_x = max(coordinates[top_left, x], coordinates[bottom_left, x])
    max_x = min(coordinates[top_right, x], coordinates[bottom_right, x])

    # y is flipped, so top and bottom are as well
    min_y = max(coordinates[top_left, y], coordinates[top_right, y])
    max_y = min(coordinates[bottom_left, y], coordinates[bottom_right, y])

    return np.array([min_y, max_y, min_x, max_x])


def get_time_from_filename(file_name):
    """Extract the time of the thumbnail. Assumes that the filename is given as [a-z]-[0-9].png"""
    pattern = '-|\.'
    splitted = re.split(pattern, file_name)
    return float(splitted[1])

extraction_args = None
def initialize_process(kwargs):
    global extraction_args
    extraction_args = kwargs


if __name__ == '__main__':
    directory = '/home/livius/Code/livius/SourceCode/Example Data/thumbnails/'

    video7_slide_coordinates = np.array([[ 0.36004776,  0.01330207],
                                         [ 0.68053395,  0.03251761],
                                         [ 0.67519468,  0.42169076],
                                         [ 0.3592881,   0.41536275]])
    video7_slide_coordinates = _inner_rectangle(video7_slide_coordinates)

    # Get all frames
    files = glob.glob(directory + '*.png')
    files.sort(key=get_time_from_filename)

    # @todo(Stephan):
    # Generate chunks that can fit into memory


    # Put everything that is constant for one video in these arguments
    args = {'slide_coordinates': video7_slide_coordinates}
    pool = Pool(processes=8, initializer=initialize_process, initargs=(args,))

    result = pool.map(extract_info, files)

    pool.close()
    pool.join()

    # result = map(lambda res: res.get(), result)

    print result

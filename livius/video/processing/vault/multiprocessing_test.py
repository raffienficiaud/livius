from multiprocessing import Pool
import glob
import cv2
import os
import json
import re
import functools

import numpy as np
import matplotlib.pyplot as plt

from ...util.histogram import get_histogram_min_max_with_percentile


def extract_lab_and_boundary(file_name):
    """Computes the Lab space of the image and the histogram_boundaries for the slide enhancement."""
    t = get_time_from_filename(file_name)

    im = cv2.imread(file_name)
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    resized_y, resized_x = im_gray.shape
    slide_crop_coordinates = extraction_args['slide_coordinates']

    min_y = slide_crop_coordinates[0] * resized_y
    max_y = slide_crop_coordinates[1] * resized_y
    min_x = slide_crop_coordinates[2] * resized_x
    max_x = slide_crop_coordinates[3] * resized_x
    slide = im_gray[min_y: max_y, min_x: max_x]
    slidehist = cv2.calcHist([slide], [0], None, [256], [0, 256])

    # Plotting the slide in order to check the slide location once
    # if t < 2:
    #     plt.subplot(2,1,1)
    #     plt.imshow(slide, cmap=plt.cm.Greys_r)
    #     plt.subplot(2,1,2)
    #     plt.plot(slidehist)
    #     plt.xlim([0,256])

    #     plt.show()

    histogram_boundaries = get_histogram_min_max_with_percentile(slidehist, False)

    # return t, histogram_boundaries
    return t, im_lab, histogram_boundaries

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


# @todo(Stephan):
# - the first and second frame of each chunk (except the first chunk) lack some information that can be computed
#   afterwards when the other chunks have been analyzed
def analyze_chunk(chunk_tuple):
    """Analyzes a chunk of files.

    :param chunk_tuple: A tuple consisting of the chunk index and the chunk itself

    :note:
       First computes the image in lab space and the boundaries for the slide enhancement.
       Then we compute the image differences and the resulting histograms for the horizontal and
       vertical stripes.

       Returns a dictionary containing all the information:
         - histogram_boundaries
         - dist_stripes
         - vert_stripes
         - energy_stripes
         - peak_stripes

       Additionally returns the last Lab image together with the corresponding hist_plane and hist_vertical_stripes
    """
    chunk_index, chunk = chunk_tuple

    # First pass
    labs = map(extract_lab_and_boundary, chunk)

    # Stripe setup
    N_stripes = 3
    N_vertical_stripes = 10

    # Loop variables
    previous_hist_plane = None
    previous_hist_vertical_stripes = None
    lab0 = None

    # If we are at a later chunk, we can re-read the corresponding frames and get the hist_planes from there
    # We cannot do this at the first chunk, because there is no image difference for the very first frame
    if chunk_index > 0:
        previous_hist_plane, previous_hist_vertical_stripes, lab0 = \
            get_hist_planes_from_image_file(chunk_index, labs[0][1], N_stripes, N_vertical_stripes)

    result_dict = {}

    for i, info in enumerate(labs):

        # Unpack information we have already computed
        t, lab, boundaries = info

        # Store the histogram boundaries for the slide enhancement
        dict_element = result_dict.get(t, {})
        result_dict[t] = dict_element
        dict_element['histogram_boundaries'] = {}
        dict_element['histogram_boundaries']['min'] = boundaries[0]
        dict_element['histogram_boundaries']['max'] = boundaries[1]

        if lab0 is not None:
            # color diff
            im_diff = (lab - lab0) ** 2
            im_diff_lab = np.sqrt(np.sum(im_diff, axis=2))

            hist_plane = horizontal_stripe_histograms(im_diff_lab, N_stripes)

            hist_vertical_stripes, energy_vertical_stripes = \
                vertical_stripe_histograms(im_diff_lab, N_vertical_stripes, do_energy_calculation=True)

            if previous_hist_plane is not None:
                dict_element['dist_stripes'] = {}
                for e, h1, h2 in zip(range(N_stripes), previous_hist_plane, hist_plane):
                    dict_element['dist_stripes'][e] = cv2.compareHist(h1, h2, cv2.cv.CV_COMP_CORREL)


            if previous_hist_vertical_stripes is not None:
                dict_element['vert_stripes'] = {}
                for e, h1, h2 in zip(range(N_vertical_stripes), previous_hist_vertical_stripes, hist_vertical_stripes):
                    dict_element['vert_stripes'][e] = cv2.compareHist(h1, h2, cv2.cv.CV_COMP_CORREL)


            # "activity" which is the enery in each stripe
            dict_element['energy_stripes'] = {}
            dict_element['peak_stripes'] = {}
            for e, energy, h1 in zip(range(N_vertical_stripes), energy_vertical_stripes, hist_vertical_stripes):
                dict_element['energy_stripes'][e] = int(energy)
                dict_element['peak_stripes'][e] = max([i for i, j in enumerate(h1) if j > 0])

            previous_hist_plane = hist_plane
            previous_hist_vertical_stripes = hist_vertical_stripes

        lab0 = lab

    return result_dict


def horizontal_stripe_histograms(im_diff_lab, N_stripes):
    # this is part of a pre-processing
    # dividing the plane vertically by N=3 and computing histograms on that. The purpose of this is to detect the environment changes
    hist_plane = []
    for i in range(N_stripes):
        location = int(i * im_diff_lab.shape[0] / float(N_stripes)), min(im_diff_lab.shape[0], int((i + 1) * im_diff_lab.shape[0] / float(N_stripes)))
        current_plane = im_diff_lab[location[0]:location[1], :]
        hist_plane.append(cv2.calcHist([current_plane.astype(np.uint8)], [0], None, [256], [0, 256]))

    return hist_plane

def vertical_stripe_histograms(im_diff_lab, N_vertical_stripes, do_energy_calculation):
    # dividing the location of the speaker by N=10 vertical stripes. The purpose of this is to detect the x location of activity/motion
    hist_vertical_stripes = []
    energy_vertical_stripes = []
    if 'speaker_bb_height_location' in extraction_args:
        speaker_bb_height_location = extraction_args['speaker_bb_height_location']
        for i in range(N_vertical_stripes):
            location = int(i * im_diff_lab.shape[1] / float(N_vertical_stripes)), min(im_diff_lab.shape[1], int((i + 1) * im_diff_lab.shape[1] / float(N_vertical_stripes)))
            current_vertical_stripe = im_diff_lab[speaker_bb_height_location[0]:speaker_bb_height_location[1], location[0]:location[1]]
            hist_vertical_stripes.append(cv2.calcHist([current_vertical_stripe.astype(np.uint8)], [0], None, [256], [0, 256]))

            if do_energy_calculation:
                energy_vertical_stripes.append(current_vertical_stripe.sum())

    if do_energy_calculation:
        return hist_vertical_stripes, energy_vertical_stripes
    else:
        return hist_vertical_stripes


def get_hist_planes_from_image_file(chunk_index, current_lab, N_stripes, N_vertical_stripes):
    im0 = cv2.imread(files[chunk_index - 1])
    im_lab0 = cv2.cvtColor(im0, cv2.COLOR_BGR2LAB)

    im_diff = (current_lab - im_lab0) ** 2
    im_diff_lab = np.sqrt(np.sum(im_diff, axis=2))

    hist_plane = horizontal_stripe_histograms(im_diff_lab, N_stripes)
    hist_vertical_stripes = vertical_stripe_histograms(im_diff_lab, N_vertical_stripes, do_energy_calculation=False)

    return hist_plane, hist_vertical_stripes, im_lab0


def chunk(l, chunk_size):
    """Splits the given list into chunks of size chunk_size. The last chunk will eventually be smaller than chunk_size."""
    result = []
    for i in range(0, len(l), chunk_size):
        result.append(l[i:i + chunk_size])

    return result


def chunk_indices(l, chunk_size):
    """Returns the indices of the beginning of each chunk"""
    return range(0, len(l), chunk_size)


def combine_dicts(dicts):
    result = {}
    map(lambda d: result.update(d), dicts)
    return result


def get_time_from_filename(file_name):
    """Extract the time of the thumbnail. Assumes that the filename is given as [a-z]-[0-9].png"""
    pattern = '-|\.'
    splitted = re.split(pattern, file_name)
    return float(splitted[1])


extraction_args = None
files = None
def initialize_process(kwargs, thumbnails):
    """Sets additional arguments that we need in order to extract information from the images"""
    global extraction_args
    global files
    extraction_args = kwargs
    files = thumbnails


if __name__ == '__main__':

    _tmp_path = '/home/livius/Code/livius/livius/Example Data/tmp'

    directory = '/home/livius/Code/livius/livius/Example Data/thumbnails/'

    video7_slide_coordinates = np.array([[0.36004776, 0.01330207],
                                         [0.68053395, 0.03251761],
                                         [0.67519468, 0.42169076],
                                         [0.3592881, 0.41536275]])
    video7_slide_coordinates = _inner_rectangle(video7_slide_coordinates)
    # @todo(Stephan): This is from a different video, pass the correct height for the resized frame of video7
    video7_speaker_bb_height_location = (155, 260)

    # Get all frames sorted by time
    files = glob.glob(directory + '*.png')
    files.sort(key=get_time_from_filename)


    chunk_size = 20
    # @todo(Stephan):
    # Check if we at any point need chunks and chunk_indices seperated,
    # otherwise we could combine them into one function
    chunks = chunk(files, chunk_size)
    chunk_ids = chunk_indices(files, chunk_size)

    # Put everything that is constant for one video in these arguments
    args = {'slide_coordinates': video7_slide_coordinates,
            'speaker_bb_height_location': video7_speaker_bb_height_location}
    pool = Pool(processes=8, initializer=initialize_process, initargs=(args, files,))

    result = pool.map(analyze_chunk, zip(chunk_ids, chunks))

    pool.close()
    pool.join()

    res = combine_dicts(result)

    with open(os.path.join(_tmp_path, 'test.json'), 'w') as f:
        json.dump(res, f, indent=2)


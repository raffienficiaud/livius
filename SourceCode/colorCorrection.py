"""Tests for determining the appropriate color correction"""

import numpy as np
import os
import cv2
import json

from util.tools import *

import matplotlib.pyplot as plt


# # Setup arrays
# hist_diffs_blue = np.empty(59)
# hist_diffs_green = np.empty(59)
# hist_diffs_red = np.empty(59)

# hist_blues = []
# hist_greens = []
# hist_reds = []

# for i in range(0,60):
# 	# Load Histograms
# 	hist_blues.append(np.load('histogram_blue' + str(i) + '.npy'))
# 	hist_greens.append(np.load('histogram_green' + str(i) + '.npy'))
# 	hist_reds.append(np.load('histogram_red' + str(i) + '.npy'))

# 	# 255 * (h - min) / (max - min)


# 	# Calculate distance
# 	if i > 0:
# 		hist_diffs_blue[i-1] = cv2.compareHist(hist_blues[i-1], hist_blues[i], cv2.cv.CV_COMP_CORREL)
# 		hist_diffs_green[i-1] = cv2.compareHist(hist_greens[i-1], hist_greens[i], cv2.cv.CV_COMP_CORREL)
# 		hist_diffs_red[i-1] = cv2.compareHist(hist_reds[i-1], hist_reds[i], cv2.cv.CV_COMP_CORREL)


# # Output
# print hist_diffs_blue
# print hist_diffs_green
# print hist_diffs_red

# plt.plot(hist_diffs_blue, color='b')
# plt.plot(hist_diffs_green, color='g')
# plt.plot(hist_diffs_red, color='r')
# plt.show()


def get_video_segments_from_histogram_correlations(histogram_correlations, tolerance, min_segment_length_in_seconds):
        """Segments the video using the histogram differences

           If there is a spike with a correlation of less than (1 - tolerance), then
           we assume that we need to adapt the histogram bounds and thus start a
           new segment.

           If there is a region with many small spikes we assume that we cannot apply
           any contrast enhancement / color correction (or apply a conservative default one).

           Returns a list of tuples marking the beginning and end of each segment
        """
        segments = []
        t_segment_start = 0.0

        lower_bounds = 1.0 - tolerance

        # @todo(Stephan):
        # This information should probably be passed together with the histogram differences
        # frames_per_histogram_differences_entry = 30
        seconds_per_correlation_entry = 1

        t = 0.0
        i = 0
        end = len(histogram_correlations)

        while i < end:

            # As long as we stay over the boundary, we count it towards the same segment
            while (i < end) and (histogram_correlations[i] >= lower_bounds):
                i += 1
                t += seconds_per_correlation_entry

            # Append segment if it is big enough
            if (t - t_segment_start) >= min_segment_length_in_seconds:
                segments.append((t_segment_start, t))

            # Skip the elements below the boundary
            while (i < end) and (histogram_correlations[i] < lower_bounds):
                i += 1
                t += seconds_per_correlation_entry

            # The new segment starts as soon as we are over the boundary again
            t_segment_start = t

        return segments

def get_histogram_boundaries_for_segment(histogram_boundaries, segment):
    """Returns the histogram boundaries for a whole segment by taking
       the average of each histogram contained in this segment."""
    start, end = segment

    histogram_bounds_in_segment = histogram_boundaries[int(start):int(end)]
    n_histograms = len(histogram_bounds_in_segment)

    min_max_sum = map(sum, zip(*histogram_bounds_in_segment))

    return (min_max_sum[0] / n_histograms, min_max_sum[1] / n_histograms)


def visualize_environment_changes_and_histogram_interpolation():

    from video.processing.postProcessing import ContrastEnhancer

    corr, boundaries, frame_ids = read_histogram_correlations_and_boundaries_from_json_file(os.path.join('Example Data', 'video7_cropped_stripe.json'))

    contrast_enhancer = ContrastEnhancer(corr, boundaries)

    segments = contrast_enhancer.segments
    segment_boundaries = contrast_enhancer.segment_histogram_boundaries


    def is_in_segment(frame, segment):
        return segment[0] <= float(frame) / 30.0 and float(frame) / 30.0 <= segment[1]

    def is_not_in_any_segment(frame):
        return not any(map(lambda seg: is_in_segment(frame, seg), segments))

    X = frame_ids

    print X

    plt.figure()

    plt.subplot(3,1,1)
    plt.plot(X, corr, 'r')
    plt.ylim([0,1])
    for (start, end) in segments:
        plt.plot(X[int(start):int(end)], corr[int(start):int(end)], 'g')

    plt.subplot(3,1,2)

    frames = range(X[0][1], X[-1][1])

    non_segment_frames = filter(is_not_in_any_segment, frames)

    min_max = map(contrast_enhancer.get_histogram_boundaries_at_time, frames)
    min_max = zip(*min_max)
    min_vals = min_max[0]
    max_vals = min_max[1]

    for (start, end) in segments:
        plt.plot(X[int(start):int(end)], min_vals[int(start):int(end)], 'b')
        plt.plot(X[int(start):int(end)], max_vals[int(start):int(end)], 'g')


    # @todo(Stephan):
    # Plot the linear interpolation

    plt.show()

    # @todo(Stephan):
    # Plot the start frames of each segment

    # plt.clf()

    # plt.subplot(1,len(segments))
    # for (start, end) in segments:
    #     plt.imshow()

    # plt.show()


visualize_environment_changes_and_histogram_interpolation()

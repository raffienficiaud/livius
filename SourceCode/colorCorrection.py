"""Tests for determining the appropriate color correction"""

import numpy as np
import os
import cv2
import json

import matplotlib.pyplot as plt

os.chdir('Example Data')
os.chdir('histograms')

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

def get_histogram_boundary_for_segment(segment, hist_bounds):
    """Returns the averaged histogram boundary for a complete segment"""
    start, end = segment

    sliced = hist_bounds[start:end]
    n_hists = len(sliced)

    summed = map(sum, zip(*sliced))

    return summed[0] / n_hists, summed[1] / n_hists

def plot_histogram_distances():
    """Reads back the sequence of histograms and plots the distance between two consecutive histograms over time"""

    with open('video7.json') as f:
        distances_histogram = json.load(f)

    frame_indices = [(i, int(i)) for i in distances_histogram.keys()]
    frame_indices.sort(key=lambda x: x[1])

    plots_dict = {}
    for count, count_integer in frame_indices:
        current_sample = distances_histogram[count]['dist_stripes']
        for i in current_sample.keys():

            if not plots_dict.has_key(int(i)):
                plots_dict[int(i)] = []

            plots_dict[int(i)].append(float(current_sample[i]))

    N_stripes = max(plots_dict.keys())

    from matplotlib import pyplot as plt



    x = frame_indices

    # Compute the Segments for the last stripe with a tolerance of 0.15
    last_stripe = plots_dict[0]

    segments = get_video_segments_from_histogram_correlations(last_stripe, tolerance=0.09, min_segment_length_in_seconds=2)
    print "Segmented Video into: ", segments


    # @todo(Stephan):
    # Use the real bounds here and test if we then can apply this to a whole segment correctly

    # fake_bounds = zip(range(len(last_stripe)), range(len(last_stripe)))
    # averaged_boundaries = get_histogram_boundary_for_segment(segments[0], fake_bounds)
    # print "Average Boundaries for fake bounds: ", averaged_boundaries

    for i in sorted(plots_dict.keys()):
        if i == 0:
            plt.title('Histgoram distance for each stripe')

        plt.subplot(N_stripes+1, 1, i+1)#, sharex=True)
        plt.plot(x, plots_dict[i], aa=False, linewidth=1)

        #lines.set_linewidth(1)
        plt.ylabel('Stripe %d' % i)

    plt.xlabel('frame #')

    plt.savefig(os.path.join(os.getcwd(), 'histogram_distance.png'))


os.chdir('../')
plot_histogram_distances()

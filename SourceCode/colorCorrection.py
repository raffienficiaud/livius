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


def visualize_environment_changes_and_histogram_interpolation(path_to_video, path_to_json_file):

    from video.processing.postProcessing import ContrastEnhancer

    # Read the Json file
    corr, boundaries, frame_ids = read_histogram_correlations_and_boundaries_from_json_file(path_to_json_file)
    X = frame_ids

    print X
    print corr
    print boundaries

    # Extract the segments and boundaries from the Contrast Enhancer
    contrast_enhancer = ContrastEnhancer(corr, boundaries)
    segments = contrast_enhancer.segments
    segment_boundaries = contrast_enhancer.segment_histogram_boundaries

    plt.figure('Histogram correlations and boundaries')

    ### Plot the histogram correlations

    subplot1 = plt.subplot(2,1,1)
    subplot1.set_title('Histogram Correlations')
    subplot1.set_ylim([0,1])

    # Plot everything red
    subplot1.plot(X, corr, 'r')

    # Plot segments green
    for (start, end) in segments:
        subplot1.plot(X[int(start):int(end)], corr[int(start):int(end)], 'g')


    ### Plot the histogram boundaries

    # Get the histogram boundary for every second
    seconds = range(len(X))
    boundaries = map(contrast_enhancer.get_histogram_boundaries_at_time, seconds)
    min_max = zip(*boundaries)
    min_values = min_max[0]
    max_values = min_max[1]

    subplot2 = plt.subplot(2,1,2)
    subplot2.set_title('Min and Max histogram boundaries used for color correction')

    def get_non_segments(frame_ids, segments):
        """Extract times that do not belong to segments"""
        non_segments = []

        # If the segments don't start at 0, we have a non-segment from 0 to the first segment
        if segments[0][0] > 0:
            non_segments.append((0, segments[0][0]))

        # Append all times between the segments
        for i in range(len(segments) - 1):
            non_segments.append((segments[i][1], segments[i+1][0]))

        # If the segments don't end at the last possible frame, we have a non-segment from the last segment to the end
        if segments[-1][1] < len(frame_ids):
            non_segments.append((segments[-1][1], len(frame_ids)))

        return non_segments

    non_segments = get_non_segments(X, segments)

    # Print Segments as straight line
    for (start, end) in segments:
        x_range = X[int(start) : int(end)]

        subplot2.plot(x_range, min_values[int(start) : int(end)], 'b')
        subplot2.plot(x_range, max_values[int(start) : int(end)], 'g')

    # Print dashes between the segments
    for (start, end) in non_segments:

        x_range = X[int(start) : int(end)]

        if len(x_range) > 1:
            # We can use dotted dash line style
            subplot2.plot(x_range, min_values[int(start) : int(end)], 'b-.')
            subplot2.plot(x_range, max_values[int(start) : int(end)], 'g-.')
        else:
            # We only have one point, draw it as a hline marker
            subplot2.plot(x_range, min_values[int(start) : int(end)], 'b_')
            subplot2.plot(x_range, max_values[int(start) : int(end)], 'g_')

    plt.show()

    plt.figure('Frames')


    ### Plot the start frames of each segment

    video = VideoFileClip(path_to_video, audio=False)

    max_pics_in_x = 4
    x_count = 1
    num_rows = (len(segments) / max_pics_in_x) + 1
    frame_count = 1


    # Adjust the segments. The Json file starts at frame 60, so we cannot take get_frame(0) for this
    # but rather get_frame(2)

    time_correction = X[0][1] / 30     # @todo(Stephan): This assumes 30fps, what to do with missing fps data?
    adjusted_segments = map(lambda (x,y): (x + time_correction, y + time_correction), segments)


    for (start, end) in adjusted_segments:
        # Extract and resize
        frame = video.get_frame(start)

        print frame.shape

        frame = cv2.resize(frame, dsize=(0,0), fx=0.1, fy=0.1)

        # Plotting
        subplot = plt.subplot(num_rows, max_pics_in_x, frame_count)
        subplot.set_title('t = ' + str(start))

        # Remove axis description
        subplot.get_xaxis().set_ticks([])
        subplot.get_yaxis().set_ticks([])

        subplot.imshow(frame)

        frame_count += 1

    plt.show()


def show_slide_summary(final_slide_clip, segments):

    summary_images = []

    COLUMNS = 7
    ROWS = 6
    resized_x = 256
    resized_y = 160

    summary_image_y = ROWS * resized_y
    summary_image_x = COLUMNS * resized_x

    def new_summary_image():
        return np.zeros((summary_image_y, summary_image_x, 3), dtype=np.uint8)

    summary_images.append(new_summary_image())

    image_count = 0

    for (start, end) in segments:
        # Extract and resize

        if image_count > COLUMNS * ROWS:
            summary_images.append(new_summary_image())

        frame = final_slide_clip.get_frame(start)
        frame = cv2.resize(frame, dsize=(resized_x, resized_y))

        summary_image = summary_images[-1]

        pos_y, pos_x = divmod(image_count, COLUMNS)

        start_x = pos_x * resized_x
        start_y = pos_y * resized_y

        end_x = start_x + resized_x
        end_y = start_y + resized_y

        summary_image[start_y : end_y, start_x : end_x, :] = np.copy(frame)

        image_count += 1


    # Show result image
    plt.figure()
    plt.imshow(summary_images[0])
    plt.show()

    return summary_images


if __name__ == '__main__':

    # visualize_environment_changes_and_histogram_interpolation(os.path.join('Example Data', 'video_7.mp4'),
    #                                                           os.path.join('Example Data', 'video7_cropped_stripe.json'))

    segments = [(0.0, 4.0), (5.0, 14.0), (15.0, 22.0), (23.0, 30.0), (31.0, 47.0), (48.0, 58.0)]

    slides = VideoFileClip(os.path.join('Example Data', 'video_7_slides.mp4'))

    show_slide_summary(slides, segments)

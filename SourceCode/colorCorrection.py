"""Tests for determining the appropriate color correction"""

import numpy as np
import os 
import cv2
import json

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

	# 255 * (h - min) / (max - min)


	# Calculate distance
	if i > 0:
		hist_diffs_blue[i-1] = cv2.compareHist(hist_blues[i-1], hist_blues[i], cv2.cv.CV_COMP_CHISQR)
		hist_diffs_green[i-1] = cv2.compareHist(hist_greens[i-1], hist_greens[i], cv2.cv.CV_COMP_CHISQR)
		hist_diffs_red[i-1] = cv2.compareHist(hist_reds[i-1], hist_reds[i], cv2.cv.CV_COMP_CHISQR)


# Output
# print hist_diffs_blue
# print hist_diffs_green
# print hist_diffs_red

# plt.plot(hist_diffs_blue, color='b')
# plt.plot(hist_diffs_green, color='g')
# plt.plot(hist_diffs_red, color='r')
# plt.show()


def get_video_segments_from_histogram_diffs(histogram_diffs, tolerance):
    """Segments the video using the histogram differences

       If the correlation betwen two histograms is less than (1 - tolerance), 
       we assume that we need to adapt the histogram bounds and thus start a
       new segment. 

       Returns a list of tuples marking the beginning and end of each segment
    """ 

    # @todo(Stephan):
    # This does not handle two consecutive changes very well. What should be the strategy here?
    # See the two datapoints:
    # 233 and 234 with the respective frame index 6990 and 7020
    # They have both correlations around 0.5 

    segments = []
    segment_start = 0
    frame_index = 0
    new_segment = False

    lower_bounds = 1.0 - tolerance
    frames_per_histogram_entry = 30

    for corr in histogram_diffs:

        # We advanced the segment_start manually, so we need to skip one comparison
        if new_segment:
            new_segment = False
            continue

        if corr < lower_bounds:
            new_segment = True
            segments.append((segment_start, frame_index))

            # Advance the segment_start, so we have inclusive boundaries in each tuple
            segment_start = frame_index + 1

        frame_index = frame_index + frames_per_histogram_entry

    # The rest of the frames build a segment as well
    if segment_start < len(histogram_diffs) * frames_per_histogram_entry:
        segments.append((segment_start, len(histogram_diffs) * frames_per_histogram_entry))

    return segments

def get_segment_histogram_boundary(segment, hist_bounds):
    """Returns the averaged histogram boundary for a complete segment"""
    start, end = segment

    sliced = hist_bounds[start:end]
    n_hists = len(sliced)

    summed = map(sum, zip(*sliced))

    return summed[0] / n_hists, summed[1] / n_hists

def plot_histogram_distances():
    """Reads back the sequence of histograms and plots the distance between two consecutive histograms over time"""

    with open('info.json') as f:
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
    last_stripe = plots_dict[2]

    segments = get_video_segments_from_histogram_diffs(last_stripe, 0.15)
    print "Segmented Video into: ", segments


    # @todo(Stephan):
    # Use the real bounds here and test if we then can apply this to a whole segment correctly 

    fake_bounds = zip(range(len(last_stripe)), range(len(last_stripe)))
    averaged_boundaries = get_segment_histogram_boundary(segments[0], fake_bounds)
    print "Average Boundaries for fake bounds: ", averaged_boundaries

    # for i in sorted(plots_dict.keys()):
    #     if i == 0:
    #         plt.title('Histgoram distance for each stripe')
        
    #     plt.subplot(N_stripes+1, 1, i+1)#, sharex=True)
    #     plt.plot(x, plots_dict[i], aa=False, linewidth=1)
        
    #     #lines.set_linewidth(1)
    #     plt.ylabel('Stripe %d' % i)
    
    # plt.xlabel('frame #')    
    
    # plt.savefig(os.path.join(os.getcwd(), 'histogram_distance.png'))
        

os.chdir('../')
plot_histogram_distances()
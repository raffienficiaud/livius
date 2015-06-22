# Import Basic modules

import numpy as np
import os

# Import everything needed to edit video clips
from moviepy.editor import *
from moviepy.Clip import *
#from moviepy.video.tools.cuts import FramesMatches
#from moviepy.video.fx.crop import crop
from moviepy.video.VideoClip import *
from moviepy.config import get_setting # ffmpeg, ffmpeg.exe, etc...

#Importing ploting libraries for shoeing purposes
import matplotlib.pyplot as plt
from pylab import *

# Importing Opencv
import cv2

class PostProcessor():
    """The PostProcessor takes care of enhancing the slide images.

       The main functionality is in the get_post_processed_slide_clip method, where the
       transformations are executed.
    """
    def __init__(self, histogram_differences):
        self.histogram_differences = histogram_differences
        self.histogram_bounds = []
        self.frame_count = 0

    def get_post_processed_slide_clip(self, clip, slide_coordinates, desiredScreenLayout=(1280,960)):
        """Computes the images of the slides and does contrast enhancement.

           This function first crops the slides from the given coordinates and warps them into perspective.

           Then we compute segments of the video according to the histogram_differences. For each of these
           segments, we compute the bounds for the contrast enhancement by averaging over all min/max values
           of all histograms contained in the segment.
           If a frame is not contained in a segment, we interpolate the min/max values between the two segments,
           it is located in.
        """

        # @todo(Stephan): Handle missing fps data.
        if clip.fps is None:
            raise

        self.fps = clip.fps

        # Warp slides into perspective
        self.clip = self.transformation3D(clip, slide_coordinates, desiredScreenLayout)

        # Histogram pass, compute bounds for the histograms of the slide images every second
        self.compute_histogram_bounds(interval_in_seconds=1)

        # Compute segments according to the histogram differences
        self.segments = self.get_video_segments_from_histogram_diffs(tolerance=0.09, min_segment_length_in_seconds=2)

        self.segment_histogram_boundaries = map(self.get_histogram_boundaries_for_segment, self.segments)

        debug = True
        if debug:
            print clip.duration
            print "Incoming Clip size", clip.size
            print "Nr of Histogram Differences:", len(self.histogram_differences)
            print "Histogram Differences:", self.histogram_differences

            print "Nr of Histogram Bounds:", len(self.histogram_bounds)
            print "Histogram Bounds:", self.histogram_bounds

            print "Nr of Computed Segments:", len(self.segments)
            print "Computed Segments:", self.segments

            print "Nr of Histogram Boundaries for the Segments:",len(self.segment_histogram_boundaries)
            print "Histogram Boundaries for the Segments", self.segment_histogram_boundaries

        # Apply the contrast enhancement
        return self.clip.fl(self.contrast_enhancement)

    def perspective_transformation2D(self, img, coordinates, desiredScreenLayout=(1280,960)):
        """Cuts out the slides from the video and warps them into perspective.
           Computes a histogram of the slides every second of the Clip.
        """
        # Extract Slides
        slideShow = np.array([[0,0],[desiredScreenLayout[0]-1,0],[desiredScreenLayout[0]-1,desiredScreenLayout[1]-1],\
                            [0,desiredScreenLayout[1]-1]],np.float32)
        retval = cv2.getPerspectiveTransform(coordinates,slideShow)
        warp = cv2.warpPerspective(img,retval,desiredScreenLayout)

        # Return slide image
        return warp

    def transformation3D(self, clip, coordinates, desiredScreenLayout=(1280,960)):
        """Applies the slide transformation to every frame"""
        def new_tranformation(frame):
            return self.perspective_transformation2D(frame, coordinates, desiredScreenLayout)

        return clip.fl_image(new_tranformation)

    def contrast_enhancement(self, get_frame, t):
        """Performs contrast enhancement by putting the colors into their full range."""
        frame = get_frame(t)

        # Retrieve histogram boundaries for this frame
        min_val, max_val = self.get_histogram_boundaries_at_time(t)

        # Perform the contrast enhancement
        framecorrected = 255.0 * (np.maximum(frame.astype(float32) - min_val, 0)) / (max_val - min_val)
        framecorrected = np.minimum(framecorrected, 255.0)
        framecorrected = framecorrected.astype(uint8)

        return framecorrected

    def get_histogram_boundaries_for_segment(self, segment):
        """Returns the histogram boundaries for a whole segment by taking
           the average of each histogram contained in this segment."""
        start, end = segment

        # @todo(Stephan): Synch the fps and the amount of histograms we look at
        start /= 30
        end /= 30

        hist_bounds_in_segment = self.histogram_bounds[start:end]
        n_hists = len(hist_bounds_in_segment)

        min_max_sum = map(sum, zip(*hist_bounds_in_segment))

        return (min_max_sum[0] / n_hists, min_max_sum[1] / n_hists)

    def get_histogram_boundaries_at_time(self, t):
        """Returns the histogram boundaries for the frame at time t.

           If this frame is inside a known segment, we already have computed the
           boundaries for it.

           If it is _not_ inside a known segment, we interpolate linearly between the
           two segments it is located.
        """

        def linear_interpolation(x, x0, x1, y0, y1):
            """Lerps x between the two points (x0, y0) and (x1, y1)"""
            return y0 + float(y1 - y0) * (float(x - x0) / float(x1 - x0))

        frame_index = int(t * self.fps)
        segment_index = 0

        debug = False
        if debug:
           print "Getting histogram_boundaries for time %s and Frame %s", (t, frame_index)

        for (start,end) in self.segments:

            if (frame_index >= start) and (frame_index <= end):
                # We are inside a segment and thus know the boundaries
                return self.segment_histogram_boundaries[segment_index]

            elif (frame_index < start):
                if segment_index == 0:
                    # @todo(Stephan):
                    # In this case, we are before the first segment.
                    # What to do here? A conservative default contrast enhancement?

                    # For now, do nothing
                    return (0, 255)

                else:
                    # We are between two segments and thus have to interpolate
                    x0 = self.segments[segment_index - 1][1]   # End of last segment
                    x1 = self.segments[segment_index][0]       # Start of new segment

                    min0, max0 = self.segment_histogram_boundaries[segment_index - 1]
                    min1, max1 = self.segment_histogram_boundaries[segment_index]

                    lerped_min = linear_interpolation(frame_index, x0, x1, min0, min1)
                    lerped_max = linear_interpolation(frame_index, x0, x1, max0, max1)

                    return (lerped_min, lerped_max)

            segment_index += 1

        # We are behind the last computed segment, since we have no end value to
        # interpolate, we just return the bounds of the last computed segment
        return self.segments[-1]

    def compute_histogram_bounds(self, interval_in_seconds=1.0):
        """Computes the histogram bounds of every interval_in_seconds of the current clip.

           Uses the 1- and 99-percentile as the approximated min/max bounds for the grayscale histogram.
        """
        # @todo(Stephan): Remove this once we get the histogram differences passed
        self.histogram_differences = []
        last_histogram = None

        # Get a frame every second
        # @todo(Stephan):
        # This seems very slow!
        # Should get better with given histogram_differences, but we still need to compute the histograms
        # on the warped (and resized) images.
        #
        # Maybe resize the image before computing the histogram? What's the performance tradeoff there?

        # for t in np.linspace(start=0, stop=self.clip.duration, num=self.clip.duration / interval_in_seconds):
        for t in xrange(int(self.clip.duration) + 1):

            frame = self.clip.get_frame(t)

            # Histogram for grayscale picture
            grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
            hist_gray = cv2.calcHist([grayscale], [0], None, [256], [0,256])

            # Normalize
            hist_gray = cv2.normalize(hist_gray)

            # Get Histogram boundaries
            bounds = PostProcessor.get_min_max_boundaries_for_normalized_histogram(hist_gray)
            self.histogram_bounds.append(bounds)

            # @todo(Stephan): Remove this once we get the histogram differences passed
            if last_histogram is not None:
                corr = cv2.compareHist(last_histogram, hist_gray, cv2.cv.CV_COMP_CORREL)
                self.histogram_differences.append(corr)
            last_histogram = hist_gray

    def get_video_segments_from_histogram_diffs(self, tolerance, min_segment_length_in_seconds):
        """Segments the video using the histogram differences

           If there is a spike with a correlation of less than (1 - tolerance), then
           we assume that we need to adapt the histogram bounds and thus start a
           new segment.

           If there is a region with many small spikes we assume that we cannot apply
           any contrast enhancement / color correction (or apply a conservative default one).

           Returns a list of tuples marking the beginning and end of each segment
        """
        segments = []
        segment_start = 0
        frame_index = 0

        lower_bounds = 1.0 - tolerance

        # @todo(Stephan):
        # This information should probably be passed together with the histogram differences
        frames_per_histogram_differences_entry = 30

        min_segment_length_in_frames = min_segment_length_in_seconds * self.fps

        i = 0
        end = len(self.histogram_differences)

        while i < end:

            # As long as we stay over the boundary, we count it towards the same segment
            while (i < end) and (self.histogram_differences[i] >= lower_bounds):
                i += 1
                frame_index += frames_per_histogram_differences_entry

            # Append segment if it is big enough
            if (frame_index - segment_start) >= min_segment_length_in_frames:
                segments.append((segment_start, frame_index))

            # Skip the elements below the boundary
            while (i < end) and (self.histogram_differences[i] < lower_bounds):
                i += 1
                frame_index += frames_per_histogram_differences_entry

            # The new segment starts as soon as we are over the boundary again
            segment_start = frame_index

        return segments

    @staticmethod
    def get_min_max_boundaries_for_normalized_histogram(hist):
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


# @todo(Stephan): Remove
# def get_histograms(clip):
#     hist_dir = os.path.join(os.getcwd(), 'histograms')

#     if not os.path.exists(hist_dir):
#         os.makedirs(hist_dir)

#     fps = clip.fps
#     frame_count = 0
#     hist_count = 0

#     for frame in clip.iter_frames():
#         if frame_count % fps == 0:
#             frame_count = 0

#             # Get Histograms of the frame for each color channel
#             hist_blue = cv2.calcHist([frame],[0],None,[256],[0,256])
#             hist_green = cv2.calcHist([frame],[1],None,[256],[0,256])
#             hist_red = cv2.calcHist([frame],[2],None,[256],[0,256])

#             # Histogram for grayscale picture
#             grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY);
#             hist_gray = cv2.calcHist([grayscale], [0], None, [256], [0,256])

#             # Normalize
#             hist_blue = cv2.normalize(hist_blue)
#             hist_green = cv2.normalize(hist_green)
#             hist_red = cv2.normalize(hist_red)
#             hist_gray = cv2.normalize(hist_gray)

#             # Obtain the estimated boundaries for each color channel
#             min_blue, max_blue = get_min_max_boundaries_for_normalized_histogram(hist_blue)
#             min_green, max_green = get_min_max_boundaries_for_normalized_histogram(hist_green)
#             min_red, max_red = get_min_max_boundaries_for_normalized_histogram(hist_red)

#             min_val = min(min_blue, min_green, min_red)
#             max_val = max(max_blue, max_green, max_red)

#             print "Min Frame :", frame[:,:,0].min()
#             print "Min Frame :", frame[:,:,1].min()
#             print "Min Frame :", frame[:,:,2].min()

#             print "min and max of each channel boundary:", min_val, max_val

#             framecorrected = 255.0 * (np.maximum(frame.astype(float32) - min_val, 0))
#             # framecorrected = 255.0 * (frame.astype(float32) - min_val)
#             framecorrected = framecorrected / (max_val - min_val)

#             framecorrected = np.minimum(framecorrected, 255.0)

#             framecorrected = framecorrected.astype(uint8)


#             # This tries out using the grayscale image

#             min_val, max_val = get_min_max_boundaries_for_normalized_histogram(hist_gray)


#             print "Min Frame :", grayscale.min()
#             print "grayscale", min_val, max_val

#             compare = 255.0 * (np.maximum(frame.astype(float32) - min_val, np.zeros(frame.shape)))
#             compare = compare / (max_val - min_val)
#             compare = np.minimum(compare, 255.0)
#             compare = compare.astype(uint8)


#             # This shows the current picture and the color correction in a window
#             fig = figure()

#             fig.add_subplot(3,1,1)
#             plt.imshow(frame)
#             plt.title("Original Frame")

#             fig.add_subplot(3,1,2)
#             plt.imshow(compare)
#             plt.title("Color correction using the 1 and 99 percentile of the grayscale histogram")

#             fig.add_subplot(3,1,3)
#             plt.imshow(framecorrected)
#             plt.title("Color correction using min and max of all color channel percentile boundaries")


#             plt.show()
#             # cv2.destroyAllWindows()

#             # framecorrected = 255.0 * (frame.astype(float32) - min_val) / (max_val - min_val)
#             # framecorrected = np.empty(frame.shape, dtype=float32)
#             # framecorrected[:,:,red] = 255.0 * (np.maximum(frame[:,:,red].astype(float32) - min_red, np.zeros(frame.shape[:2]))) / (max_red - min_red)
#             # framecorrected[:,:,green] = 255.0 * (np.maximum(frame[:,:,green].astype(float32) - min_green, np.zeros(frame.shape[:2]))) / (max_green - min_green)
#             # framecorrected[:,:,blue] = 255.0 * (np.maximum(frame[:,:,blue].astype(float32) - min_blue, np.zeros(frame.shape[:2]))) / (max_blue - min_blue)



#             # Save Histogram for each color channel
#             np.save('histograms/histogram_blue' + str(hist_count), hist_blue)
#             np.save('histograms/histogram_green' + str(hist_count), hist_green)
#             np.save('histograms/histogram_red' + str(hist_count), hist_red)


#             # Plot and save as png

#             # plt.plot(hist_blue, color='b')
#             # plt.plot(hist_green, color='g')
#             # plt.plot(hist_red, color='r')
#             # plt.xlim([0,256])

#             # plt.savefig('histograms/histogram' + str(hist_count) + '.png')
#             # plt.clf()

#             # plt.plot(hist_gray)
#             # plt.xlim([0,256])
#             # plt.savefig('histograms/histogram_gray' + str(hist_count) + '.png')
#             # plt.clf()

#             hist_count += 1

#         frame_count += 1


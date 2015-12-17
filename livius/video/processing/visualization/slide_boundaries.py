"""
Visualise the min/max function for the slides contrast


"""



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

    # ## Plot the histogram correlations

    subplot1 = plt.subplot(2, 1, 1)
    subplot1.set_title('Histogram Correlations')
    subplot1.set_ylim([0, 1])

    # Plot everything red
    subplot1.plot(X, corr, 'r')

    # Plot segments green
    for (start, end) in segments:
        subplot1.plot(X[int(start):int(end)], corr[int(start):int(end)], 'g')


    # ## Plot the histogram boundaries

    # Get the histogram boundary for every second
    seconds = range(len(X))
    boundaries = map(contrast_enhancer.get_histogram_boundaries_at_time, seconds)
    min_max = zip(*boundaries)
    min_values = min_max[0]
    max_values = min_max[1]

    subplot2 = plt.subplot(2, 1, 2)
    subplot2.set_title('Min and Max histogram boundaries used for color correction')

    def get_non_segments(frame_ids, segments):
        """Extract times that do not belong to segments"""
        non_segments = []

        # If the segments don't start at 0, we have a non-segment from 0 to the first segment
        if segments[0][0] > 0:
            non_segments.append((0, segments[0][0]))

        # Append all times between the segments
        for i in range(len(segments) - 1):
            non_segments.append((segments[i][1], segments[i + 1][0]))

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


    # ## Plot the start frames of each segment

    video = VideoFileClip(path_to_video, audio=False)

    max_pics_in_x = 4
    x_count = 1
    num_rows = (len(segments) / max_pics_in_x) + 1
    frame_count = 1


    # Adjust the segments. The Json file starts at frame 60, so we cannot take get_frame(0) for this
    # but rather get_frame(2)

    time_correction = X[0][1] / 30  # @todo(Stephan): This assumes 30fps, what to do with missing fps data?
    adjusted_segments = map(lambda (x, y): (x + time_correction, y + time_correction), segments)


    for (start, end) in adjusted_segments:
        # Extract and resize
        frame = video.get_frame(start)

        print frame.shape

        frame = cv2.resize(frame, dsize=(0, 0), fx=0.1, fy=0.1)

        # Plotting
        subplot = plt.subplot(num_rows, max_pics_in_x, frame_count)
        subplot.set_title('t = ' + str(start))

        # Remove axis description
        subplot.get_xaxis().set_ticks([])
        subplot.get_yaxis().set_ticks([])

        subplot.imshow(frame)

        frame_count += 1

    plt.show()

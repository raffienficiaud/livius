"""
Creates summary of the slide detection/extraction


"""


def get_slide_summary(final_slide_clip, segments, output_shape=(7, 6), slide_size=(256, 160)):
    """Gets the first frame of each computed segment and saves it to one or more summary images.

       Returns an array of all the summary images.
    """

    summary_images = []

    columns = output_shape[0]
    rows = output_shape[1]
    resized_x = slide_size[0]
    resized_y = slide_size[1]

    summary_image_y = rows * resized_y
    summary_image_x = columns * resized_x

    def new_summary_image():
        return np.zeros((summary_image_y, summary_image_x, 3), dtype=np.uint8)

    summary_images.append(new_summary_image())

    image_count = 0
    summary_count = 1

    for (start, end) in segments:
        if image_count >= columns * rows:
            summary_images.append(new_summary_image())
            image_count = 0

        # Extract and resize frame
        frame = final_slide_clip.get_frame(start)
        frame = cv2.resize(frame, dsize=(resized_x, resized_y))

        summary_image = summary_images[-1]

        # Determine position in image
        pos_y, pos_x = divmod(image_count, columns)

        start_x = pos_x * resized_x
        start_y = pos_y * resized_y

        end_x = start_x + resized_x
        end_y = start_y + resized_y

        # Copy resized frame to summary image
        summary_image[start_y : end_y, start_x : end_x, :] = np.copy(frame)

        image_count += 1

    return summary_images

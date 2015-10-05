"""
Layout
======

This module implements layout and video formation for video editing part
main function:

.. autosummary::

  createFinalVideo

"""

import sys
from moviepy.video.compositing.concatenate import concatenate

import logging
logger = logging.getLogger()

# this settings are dependent on the background image

#: Default layout
default_layout = {'canvas_video_size': (1920, 1080),
                  'slides_video_size': (1280, 960),
                  'speaker_video_size': (620, 360),
                  'speaker_video_position': (0, 360),
                  'slides_video_position': (640, 60)}


def createFinalVideo(slide_clip,
                     speaker_clip,
                     audio_clip,
                     video_background_image,
                     intro_image_and_durations,
                     credit_images_and_durations,
                     fps,
                     layout=None,
                     talkInfo=' ',
                     speakerInfo=' ',
                     dateInfo='',
                     first_segment_duration=10,
                     output_file_name='Output',
                     codecFormat='libx264',
                     container='.mp4',
                     flagWrite=True):

    """
    This function serves to form the video layout, create the final video and write it.

    :param slide_clip: The slide clip. This is a moviepy video object.
    :param speaker_clip: The screen clip. This is a moviepy video object.
    :param str video_background_image: the path to background image
    :param list credit_images_and_durations: a list of 2-uples containing the credit information. The first element
      of the tuple is the image file to be shozn, the second is the duration of the image in the video.
      If the duration is ``None``, it is set to 2 seconds.
    :param audio_clip: the audio_clip file to be attached to the video file. This is a moviepy audio_clip object.
    :param fps: frame per second
    :param dict layout: indicates the layout of the different video streams. This is a dictionary containing the
      following elements, each of them being a 2-uple indicating either a size or a position:

      * "canvas_video_size": The desired size of whole layout
      * "slides_video_size": The desired size of screen part
      * "speaker_video_size": The desired size of speaker part
      * "speaker_video_position": the position of the speaker substream
      * "slides_video_position": the position of the slides substream

      If any of those parameter is missing, it is replaced by the default given by the default layout.

      .. note: this layout is of course dependent on the background image

    :param str talkInfo: Information about talk like the title, the subject, etc.
    :param str speakerInfo: Information about talk like the title, the subject, etc.
    :param str dateInfo: Information about the date of talk.
    :param first_segment_duration: Duration *in seconds* of the first segment of the video, showing the title.
        Defaults to 10 seconds.
    :param str output_file_name: the output file name without extension.
    :param str codecFormat: The codec format to write the new video into the file
        Codec to use for image encoding. Can be any codec supported by ffmpeg, but the container
        must be set accordingly.
    :param str container: the video file format (container) including all streams.
    :param bool flagWrite: A flag to set whether write the new video or not

    .. rubric:: Images

    The background image is shown during the whole video.

    .. rubric:: Hints for video

    You may simply change the codecs, fps and etc inside the function.

    Video codecs:

    * ``'libx264'``: (use file extension ``.mp4``)
      makes well-compressed videos (quality tunable using 'bitrate').
    * ``'mpeg4'``: (use file extension ``.mp4``) can be an alternative
      to ``'libx264'``, and produces higher quality videos by default.
    * ``'rawvideo'``: (use file extension ``.avi``) will produce
      a video of perfect quality, of possibly very huge size.
    * ``png``: (use file extension ``.avi``) will produce a video
      of perfect quality, of smaller size than with ``rawvideo``
    * ``'libvorbis'``: (use file extension ``.ogv``) is a nice video
      format, which is completely free/ open source. However not
      everyone has the codecs installed by default on their machine.
    * ``'libvpx'``: (use file extension ``.webm``) is tiny a video
      format well indicated for web videos (with HTML5). Open source.


    .. rubric:: Hints for audio

    The parameter ``audio_clip`` may be a boolean or the path of a file.
    If ``True`` and the clip has an audio_clip clip attached, this
    audio_clip clip will be incorporated as a soundtrack in the movie.
    If ``audio_clip`` is the name of an audio_clip file, this audio_clip file
    will be incorporated as a soundtrack in the movie.

    Possible audio_clip codecs are:

    * ``'libmp3lame'``: for '.mp3'
    * ``'libvorbis'``: for 'ogg'
    * ``'libfdk_aac'``: for 'm4a',
    * ``'pcm_s16le'``: for 16-bit wav
    * ``'pcm_s32le'``: for 32-bit wav.

     Example::

        createFinalVideo(slide_clip,speaker_clip,
                        video_background_image,
                        audio_clip,
                        credit_images_and_durations,
                        fps = 30,
                        canvas_video_size = (1920, 1080),
                        slides_video_size = (1280, 960),
                        speaker_video_size = (620, 360),
                        talkInfo = 'How to use SVM kernels',
                        speakerInfo = 'Prof. Bernhard Schoelkopf',
                        dateInfo = 'July 2015',
                        first_segment_duration = 10,
                        output_file_name = 'video',
                        codecFormat = 'libx264',
                        container = '.mp4',
                        flagWrite = True)


    """

    from moviepy.video.VideoClip import ImageClip, TextClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

    # setting the sizes
    if layout is None:
        layout = {}

    final_layout = dict(default_layout)
    final_layout.update(layout)

    canvas_video_size = final_layout['canvas_video_size']
    slides_video_size = final_layout['slides_video_size']
    speaker_video_size = final_layout['speaker_video_size']
    speaker_video_position = final_layout['speaker_video_position']
    slides_video_position = final_layout['slides_video_position']

    # some utility functions
    def resize_clip_if_needed(clip, desired_size):
        if clip.w != desired_size[0] or clip.h != desired_size[1] or True:
            return clip.resize(desired_size)
        return clip

    def create_image_clip(image_filename):
        # having some encoding issues in moviepy
        if isinstance(image_filename, unicode):
            image_filename = image_filename.encode(sys.getdefaultencoding())

        return ImageClip(image_filename)

    def create_slide_show_of_images(images_and_durations):
        all_slides = []
        for image, duration in images_and_durations:

            image_clip = create_image_clip(image)
            if (image_clip.w != canvas_video_size[0]) or (image_clip.h != canvas_video_size[1]):
                # resize only if needed
                image_clip = image_clip.resize((canvas_video_size[1], canvas_video_size[1]))

            image_clip = image_clip.set_duration(duration if duration is not None else 2)  # 2 secs is the default

            all_slides.append(image_clip)

        return concatenate(all_slides)

    if isinstance(video_background_image, unicode):
        video_background_image = video_background_image.encode(sys.getdefaultencoding())

    if (canvas_video_size[0] < (slides_video_size[0] or speaker_video_size[0])) and \
       (canvas_video_size[1] < (slides_video_size[1] or speaker_video_size[1])):

        logger.warning("[layout] Warning: The selected sizes are not appropriate")


    import ipdb
    ipdb.set_trace()




    ####
    # First segment: title,
    if intro_image_and_durations is None or not intro_image_and_durations:
        # this is shitty, we do not use that
        # The template for the second show
        instituteText = "Max Planck Institute for Intelligent Systems"
        defaultText = 'Machine Learning Summer School 2015'

        left_margin = int(canvas_video_size[0] / 3) - 100
        right_margin = 100
        width = canvas_video_size[0] - left_margin - right_margin

        pixelPerCharTitleText = width // len(talkInfo)
        pixelPerCharSpeakerText = width // len(speakerInfo)

        txtClipTitleText = TextClip(talkInfo, fontsize=pixelPerCharTitleText, color='white', font="Amiri")
        txtClipTitleText = txtClipTitleText.set_position((left_margin, canvas_video_size[1] / 3))

        txtClipSpeakerText = TextClip(speakerInfo, fontsize=pixelPerCharSpeakerText, color='white', font="Amiri")
        txtClipSpeakerText = txtClipSpeakerText.set_position((left_margin, canvas_video_size[1] / 3 + 100))

        txtClipInstituteText = TextClip(instituteText, fontsize=36, color='white', font="Amiri")
        txtClipInstituteText = txtClipInstituteText.set_position((left_margin, canvas_video_size[1] / 3 + 170))

        txtClipDefaultText = TextClip(defaultText, fontsize=40, color='white', font="Amiri")
        txtClipDefaultText = txtClipDefaultText.set_position((left_margin, canvas_video_size[1] / 3 + 300))

        txtClipDateText = TextClip(dateInfo, fontsize=40, color='white', font="Amiri")
        txtClipDateText = txtClipDateText.set_position((left_margin, canvas_video_size[1] / 3 + 350))

        # this does not work, the sizes should be set properly
        # and the fonts are ugly
        first_segment_clip = CompositeVideoClip([txtClipTitleText])  #, txtClipSpeakerText, txtClipInstituteText, txtClipDefaultText, txtClipDateText])
        first_segment_clip = first_segment_clip.set_duration(first_segment_duration)
        #first_segment_clip = first_segment_clip.set_start(0)

    else:
        first_segment_clip = create_slide_show_of_images(intro_image_and_durations)

    ####
    # second segment: the slides, videos, audio_clip, etc.
    # resizing the slides and speaker clips if needed
    speaker_clip_composed = resize_clip_if_needed(speaker_clip, speaker_video_size).set_position(speaker_video_position)
    # the audio_clip is associated to this clip
    if audio_clip is not None:
        speaker_clip_composed = speaker_clip_composed.set_audio(audio_clip)

    slide_clip_composed = resize_clip_if_needed(slide_clip, slides_video_size).set_position(slides_video_position)

    # this one will be used for reference on properties
    # apparently in MoviePy, we cannot nest CompositeVideoClip, which is why it is done this way
    # and not putting this part into a CompositeVideoClip
    second_segment_clip = slide_clip_composed


    second_segment_clip = second_segment_clip.set_duration(5)  #speaker_clip.duration)  # same duration as any input clip

    #! TODO
    # if we need to pause things, and to set start and stop, this should be done here

    #second_segment_clip = second_segment_clip.set_start(first_segment_clip.end)

    ####
    # second segment overlay: title, info, background: duration equal to the second segment clip
    info_underslides = '%s - %s' % (speakerInfo, talkInfo)
    talk_info_clip = TextClip(info_underslides, fontsize=30, color='white', font="Amiri")

    talk_info_clip = talk_info_clip.set_position((slide_clip_composed.pos(0)[0],
                                                  slide_clip_composed.pos(0)[1] + slide_clip_composed.h + 15))

    background_image_clip = resize_clip_if_needed(create_image_clip(video_background_image), canvas_video_size)

    second_segment_overlay_clip = CompositeVideoClip([background_image_clip,
                                                      talk_info_clip,
                                                      speaker_clip_composed,
                                                      slide_clip_composed])

    # same attributes as the clip it is supposed to overlay
    second_segment_overlay_clip = second_segment_overlay_clip.set_duration(second_segment_clip.duration)

    ###
    # third segment: credits etc.
    third_segment_clip = create_slide_show_of_images(credit_images_and_durations)

    # the final video
    outputVideo = concatenate([first_segment_clip, second_segment_overlay_clip, third_segment_clip])

    if flagWrite:
        outputVideo.write_videofile(output_file_name + container, fps, codec=codecFormat)

    return outputVideo

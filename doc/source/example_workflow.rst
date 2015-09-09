==================
A Workflow Example
==================

This page describes how to build and run a workflow (= DAG of Jobs)

------------
Introduction
------------

In the :py:mod:`workflow module <SourceCode.video.processing.workflow>`  there are two example workflows that can be used. One only extracts
the thumbnails from the video and the other one.

The workflow can be run from the command line in the main **livius** folder via

    python -m SourceCode.video.processing.workflow

(after following the livius installation procedure from Parnia's documentation.)

--------------------------------------
Workflow for extracting the Slide Clip
--------------------------------------

The creation of the workflow is as follows ::

    def workflow_extract_slide_clip():
        """
        Return a workflow that extracs the slide clip from a video.

        Consists of many tasks such as
            * ffmpg thumbnail generation
            * Polygon Selection for the Slides and Speaker
            * Histogram Computations
            * Histogram Correlations
            * Segment Computation
            * Perspective Transformations and Contrast Enhancement.
        """
        HistogramsLABDiff.add_parent(GatherSelections)
        HistogramsLABDiff.add_parent(FFMpegThumbnailsJob)

        HistogramCorrelationJob.add_parent(HistogramsLABDiff)
        HistogramCorrelationJob.add_parent(NumberOfFilesJob)

        SegmentComputationJob.add_parent(HistogramCorrelationJob)
        SegmentComputationJob.add_parent(NumberOfFilesJob)

        ContrastEnhancementBoundaries.add_parent(FFMpegThumbnailsJob)
        ContrastEnhancementBoundaries.add_parent(SelectSlide)
        ContrastEnhancementBoundaries.add_parent(SegmentComputationJob)

        return ExtractSlideClipJob


The ``__main__`` function of the Script shows how to run a specific workflow. ::

    if __name__ == '__main__':

        [...]

        # Set up the folders for processing
        root_folder = os.path.join(os.path.dirname(__file__),
                                   os.pardir,
                                   os.pardir,
                                   os.pardir)
        video_folder = os.path.join(root_folder, 'Videos')
        current_video = os.path.join(video_folder, 'video_7.mp4')
        proc_folder = os.path.abspath(os.path.join(root_folder, 'tmp'))
        slide_clip_folder = os.path.abspath(os.path.join(proc_folder, 'Slide Clip'))

        if not os.path.exists(proc_folder):
            os.makedirs(proc_folder)

        if not os.path.exists(slide_clip_folder):
            os.makedirs(slide_clip_folder)

        # Create the workflow
        workflow = workflow_extract_slide_clip()

        # Specify all parameters
        params = {'video_filename': current_video,
                  'thumbnails_location': os.path.join(proc_folder, 'processing_video_7_thumbnails'),
                  'json_prefix': os.path.join(proc_folder, 'processing_video_7_'),
                  'segment_computation_tolerance': 0.05,
                  'segment_computation_min_length_in_seconds': 2,
                  'slide_clip_desired_format': [1280, 960],
                  'nb_vertical_stripes': 10}

        # Process the workflow
        outputs = process(workflow, **params)

        # The outputs of the workflow is a Moviepy VideoClip object
        # Write it to disk
        outputs.write_videofile(os.path.join(slide_clip_folder, 'slideclip.mp4'))


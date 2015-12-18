'''
User interaction
================

Module containing several functions for interacting with users.
'''

import cv2
import numpy as np
from random import randint


def get_polygon_from_user(im,
                          nb_points_polygon,
                          window_name=None):
    """
    Shows a window with the given image, inviting the user to click on several points
    in order to define a polygon

    :param im: image to show as a numpy array
    :param nb_points_polygon: number of points of the polygon
    :param windows_name: the title of the window shown to the user. If None, a random string will be shown

    If the user pressed the ESC key during the selection, the method returns with an empty list.

    """

    if window_name is None:
        window_name = ''.join([chr(ord('a') + randint(0, 26)) for _ in range(10)])

    params = type('params', (object,), {})()
    params.current_position = (0, 0)
    params.click = None
    params.click_position = None
    params.points = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1200)
    cv2.moveWindow(window_name, 100, 100)

    def onMouse(event, x, y, flags, param):

        param.current_position = (x, y)

        if not (flags & cv2.EVENT_FLAG_LBUTTON) and params.click:
            params.click = False
            params.points.append(params.click_position)
            params.click_position = None

        if flags & cv2.EVENT_FLAG_LBUTTON and params.click_position is None:
            params.click = True
            params.click_position = (x, y)

    cv2.setMouseCallback(window_name, onMouse, params)

    while len(params.points) < nb_points_polygon:

        im_draw = np.copy(im)
        if len(params.points) > 1:
            for index in range(1, len(params.points)):
                cv2.line(im_draw, params.points[index - 1], params.points[index], (255, 0, 0))

        if len(params.points) > 0 and params.current_position is not None:
            cv2.line(im_draw, params.points[-1], params.current_position, (255, 0, 0))

        cv2.line(im_draw, (params.current_position[0], 0), (params.current_position[0], im_draw.shape[0]), (0, 255, 0))
        cv2.line(im_draw, (0, params.current_position[1]), (im_draw.shape[1], params.current_position[1]), (0, 255, 0))
        cv2.imshow(window_name, im_draw)

        wk = cv2.waitKey(10)
        if (wk & 255) == 27:  # escape key
            cv2.destroyWindow(window_name)
            cv2.waitKey(1000)
            params.points = []
            break

    cv2.destroyWindow(window_name)

    return params.points


class GetPolygon(object):
    """Same as :py:func:`get_polygon_from_user` but allows for the image update without creating a new window"""

    @staticmethod
    def onMouse(event, x, y, flags, params):

        if params is None:
            return

        params.current_position = (x, y)

        if not (flags & cv2.EVENT_FLAG_LBUTTON) and params.click:
            params.click = False
            params.points.append(params.click_position)
            params.click_position = None

        if flags & cv2.EVENT_FLAG_LBUTTON and params.click_position is None:
            params.click = True
            params.click_position = (x, y)

    def __init__(self, window_name=None):

        self.window_name = window_name
        if self.window_name is None:
            self.window_name = ''.join([chr(ord('a') + randint(0, 26)) for _ in range(10)])

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1200)
        cv2.moveWindow(self.window_name, 100, 100)


    def select_polygon_on_image(self, im, nb_points_polygon):

        params = type('params', (object,), {})()
        params.current_position = (0, 0)
        params.click = None
        params.click_position = None
        params.points = []

        cv2.setMouseCallback(self.window_name, self.onMouse, params)

        while len(params.points) < nb_points_polygon:

            im_draw = np.copy(im)
            if len(params.points) > 1:
                for index in range(1, len(params.points)):
                    cv2.line(im_draw, params.points[index - 1], params.points[index], (255, 0, 0))

            if len(params.points) > 0 and params.current_position is not None:
                cv2.line(im_draw, params.points[-1], params.current_position, (255, 0, 0))

            cv2.line(im_draw, (params.current_position[0], 0), (params.current_position[0], im_draw.shape[0]), (0, 255, 0))
            cv2.line(im_draw, (0, params.current_position[1]), (im_draw.shape[1], params.current_position[1]), (0, 255, 0))
            cv2.imshow(self.window_name, im_draw)

            wk = cv2.waitKey(10)
            if (wk & 255) == 27:  # escape key
                params.points = []
                break

        cv2.setMouseCallback(self.window_name, self.onMouse, None)
        return params.points

    def __del__(self):
        if self.window_name is not None:
            self.close()

    def close(self):
        cv2.destroyWindow(self.window_name)
        self.window_name = None

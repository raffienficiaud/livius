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
    """Shows a window with the given image, inviting the user to click on several points
    in order to define a polygon

    :param (numpy array) im: image to show
    :param nb_points_polygon: number of points of the polygon
    :param windows_name: the title of the window shown to the user. If None, a random string will be shown

    """

    if window_name is None:
        window_name = ''.join([chr(ord('a') + randint(0, 26)) for _ in range(10)])

    params = type('params', (object,), {})()
    params.current_position = None
    params.click = None
    params.click_position = None
    params.points = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 100, 100)

    def onMouse(event, x, y, flags, param):

        param.current_position = (x, y)

        if not (flags & cv2.EVENT_FLAG_LBUTTON) and params.click:
            params.click = False
            params.points.append(params.click_position)
            print params.points
            params.click_position = None

        if flags & cv2.EVENT_FLAG_LBUTTON and params.click_position is None:
            params.click = True
            params.click_position = (x, y)

    cv2.setMouseCallback(window_name, onMouse, params)
    cv2.imshow(window_name, im)

    points = params.points
    while len(params.points) < 4:

        im_draw = np.copy(im)
        if len(points) > 1:
            for index in range(1, len(points)):
                cv2.line(im_draw, points[index - 1], points[index], (255, 0, 0))

        if len(points) > 0 and params.current_position is not None:
            cv2.line(im_draw, points[-1], params.current_position, (255, 0, 0))

        cv2.imshow(window_name, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(window_name)

    return points

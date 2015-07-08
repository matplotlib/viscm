# coding=utf8
# BézierBuilder
#
# Copyright (c) 2013, Juan Luis Cano Rodríguez <juanlu001@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""BézierBuilder, an interactive Bézier curve explorer.

Just run it with

$ python bezier_builder.py

"""

import numpy as np
from scipy.special import binom

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .minimvc import Trigger

class BezierModel(object):
    def __init__(self, xp, yp):
        self._xp = list(xp)
        self._yp = list(yp)

        self.trigger = Trigger()

    def get_control_points(self):
        return list(self._xp), list(self._yp)

    def get_bezier_points(self, num=200):
        return self.get_bezier_points_at(np.linspace(0, 1, num))

    def get_bezier_points_at(self, at, grid=256):
        at = np.asarray(at)
        # The Bezier curve is parameterized by a value t which ranges from 0
        # to 1. However, there is a nonlinear relationship between this value
        # and arclength. We want to parameterize by t', which measures
        # normalized arclength. To do this, we have to calculate the function
        # arclength(t), and then invert it.
        t = np.linspace(0, 1, grid)
        x, y = Bezier(list(zip(self._xp, self._yp)), t).T
        x_deltas = np.diff(x)
        y_deltas = np.diff(y)
        arclength_deltas = np.empty(t.shape)
        arclength_deltas[0] = 0
        np.hypot(x_deltas, y_deltas, out=arclength_deltas[1:])
        arclength = np.cumsum(arclength_deltas)
        arclength /= arclength[-1]
        # Now (t, arclength) is a LUT describing the t -> arclength mapping
        # Invert it to get at -> t
        at_t = np.interp(at, arclength, t)
        # And finally look up at the Bezier values at at_t
        # (Might be quicker to np.interp againts x and y, but eh, doesn't
        # really matter.)
        return Bezier(list(zip(self._xp, self._yp)), at_t).T

    def add_point(self, i, new_x, new_y):
        self._xp.insert(i, new_x)
        self._yp.insert(i, new_y)
        self.trigger.fire()

    def remove_point(self, i):
        del self._xp[i]
        del self._yp[i]
        self.trigger.fire()

    def move_point(self, i, new_x, new_y):
        self._xp[i] = new_x
        self._yp[i] = new_y
        self.trigger.fire()

    def set_control_points(self, xp, yp):
        self._xp = list(xp)
        self._yp = list(yp)
        self.trigger.fire()

class BezierBuilder(object):
    """Bézier curve interactive builder.

    """
    def __init__(self, ax, bezier_model):
        self.ax = ax
        self.bezier_model = bezier_model

        self.canvas = self.ax.figure.canvas
        xp, yp = self.bezier_model.get_control_points()
        self.control_polygon = Line2D(xp, yp,
                                      ls="--", c="#666666", marker="x",
                                      mew=2, mec="#204a87")
        self.ax.add_line(self.control_polygon)
        x, y = self.bezier_model.get_bezier_points()
        self.bezier_curve = Line2D(x, y)
        self.ax.add_line(self.bezier_curve)

        # Event handler for mouse clicking
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_notify)

        self._index = None  # Active vertex

        self.bezier_model.trigger.add_callback(self._refresh)
        self._refresh()

    def __del__(self):
        self.bezier_model.trigger.remove_callback(self._refresh)

    def on_button_press(self, event):
        # Ignore clicks outside axes
        if event.inaxes != self.ax: return
        res, ind = self.control_polygon.contains(event)
        if res and event.key is None:
            # Grabbing a point to drag
            self._index = ind["ind"][0]
        if res and event.key == "control":
            # Control-click deletes
            self.bezier_model.remove_point(ind["ind"][0])
        if event.key == "shift":
            # Adding a new point. Find the two closest points and insert it in
            # between them.
            total_squared_dists = []
            xp, yp = self.bezier_model.get_control_points()
            for i in range(len(xp) - 1):
                dist = (event.xdata - xp[i]) ** 2
                dist += (event.ydata - yp[i]) ** 2
                dist += (event.xdata - xp[i + 1]) ** 2
                dist += (event.ydata - yp[i + 1]) ** 2
                total_squared_dists.append(dist)
            best = np.argmin(total_squared_dists)

            self.bezier_model.add_point(best + 1, event.xdata, event.ydata)

    def on_button_release(self, event):
        if event.button != 1: return
        self._index = None

    def on_motion_notify(self, event):
        if event.inaxes != self.ax: return
        if self._index is None: return
        x, y = event.xdata, event.ydata

        self.bezier_model.move_point(self._index, x, y)

    def _refresh(self):
        xp, yp = self.bezier_model.get_control_points()
        self.control_polygon.set_data(xp, yp)
        x, y = self.bezier_model.get_bezier_points()
        self.bezier_curve.set_data(x, y)
        self.canvas.draw()

def Bernstein(n, k):
    """Bernstein polynomial.

    """
    coeff = binom(n, k)

    def _bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return _bpoly


def Bezier(points, at):
    """Build Bézier curve from points.

    """
    at = np.asarray(at)
    at_flat = at.ravel()
    N = len(points)
    curve = np.zeros((at_flat.shape[0], 2))
    for ii in range(N):
        curve += np.outer(Bernstein(N - 1, ii)(at_flat), points[ii])
    return curve.reshape(at.shape + (2,))

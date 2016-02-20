# This file is part of viscm
# Copyright (C) 2015 Nathaniel Smith <njs@pobox.com>
# Copyright (C) 2015 Stefan van der Walt <stefanv@berkeley.edu>
# See file LICENSE.txt for license information.

# Simple script using CIECAM02 and CAM02-UCS to visualize properties of a
# matplotlib colormap

import sys
import os.path

import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap

from colorspacious import (cspace_converter, cspace_convert,
                           CIECAM02Space, CIECAM02Surround, CAM02UCS)
from .minimvc import Trigger

# The correct L_A value for the standard sRGB viewing conditions is:
#   (64 / np.pi) / 5
# Due to an error in our color conversion code, the matplotlib colormaps were
# designed using the assumption that they would be viewed with an L_A value of
#   (64 / np.pi) * 5
# (i.e., 125x brighter ambient illumination than appropriate). It turns out
# that when all is said and done this has negligible effect on the uniformity
# of the resulting colormaps (phew), BUT fixing the bug has the effect of
# somewhat shrinking the sRGB color solid as projected into CAM02-UCS
# space. This means that the bezier points for existing colormaps (like the
# matplotlib ones) are in the wrong place. We can reproduce the original
# colormaps from these points by using this buggy_CAM02UCS space as our
# uniform space:
buggy_sRGB_viewing_conditions = CIECAM02Space(
    XYZ100_w="D65",
    Y_b=20,
    L_A=(64 / np.pi) * 5,  # bug: should be / 5
    surround=CIECAM02Surround.AVERAGE)
buggy_CAM02UCS = {"name": "CAM02-UCS",
                  "ciecam02_space": buggy_sRGB_viewing_conditions,
                  }

GREYSCALE_CONVERSION_SPACE = "JCh"

_sRGB1_to_JCh = cspace_converter("sRGB1", GREYSCALE_CONVERSION_SPACE)
_JCh_to_sRGB1 = cspace_converter(GREYSCALE_CONVERSION_SPACE, "sRGB1")
def to_greyscale(sRGB1):
    JCh = _sRGB1_to_JCh(sRGB1)
    JCh[..., 1] = 0
    return _JCh_to_sRGB1(JCh)

_deuter50_space = {"name": "sRGB1+CVD",
                   "cvd_type": "deuteranomaly",
                   "severity": 50}
_deuter50_to_sRGB1 = cspace_converter(_deuter50_space, "sRGB1")
_deuter100_space = {"name": "sRGB1+CVD",
                    "cvd_type": "deuteranomaly",
                    "severity": 100}
_deuter100_to_sRGB1 = cspace_converter(_deuter100_space, "sRGB1")
_prot50_space = {"name": "sRGB1+CVD",
                 "cvd_type": "protanomaly",
                 "severity": 50}
_prot50_to_sRGB1 = cspace_converter(_prot50_space, "sRGB1")
_prot100_space = {"name": "sRGB1+CVD",
                  "cvd_type": "protanomaly",
                  "severity": 100}
_prot100_to_sRGB1 = cspace_converter(_prot100_space, "sRGB1")

def _show_cmap(ax, rgb):
    ax.imshow(rgb[np.newaxis, ...], aspect="auto")

def _apply_rgb_mat(mat, rgb):
    return np.clip(np.einsum("...ij,...j->...i", mat, rgb), 0, 1)

# sRGB corners: a' goes from -37.4 to 45
AP_LIM = (-38, 46)
# b' goes from -46.5 to 42
BP_LIM = (-47, 43)
# J'/K goes from 0 to 100
JP_LIM = (-1, 101)


def _setup_Jpapbp_axis(ax):
    ax.set_xlabel("a' (green -> red)")
    ax.set_ylabel("b' (blue -> yellow)")
    ax.set_zlabel("J'/K (white -> black)")
    ax.set_xlim(*AP_LIM)
    ax.set_ylim(*BP_LIM)
    ax.set_zlim(*JP_LIM)


# Adapt a matplotlib colormap to a linearly transformed version -- useful for
# visualizing how colormaps look given color deficiency.
# Kinda a hack, b/c we inherit from Colormap (this is required), but then
# ignore its implementation entirely.
class TransformedCMap(matplotlib.colors.Colormap):
    def __init__(self, transform, base_cmap):
        self.transform = transform
        self.base_cmap = base_cmap

    def __call__(self, *args, **kwargs):
        fx = self.base_cmap(*args, **kwargs)
        tfx = self.transform(fx)
        return tfx

    def set_bad(self, *args, **kwargs):
        self.base_cmap.set_bad(*args, **kwargs)

    def set_under(self, *args, **kwargs):
        self.base_cmap.set_under(*args, **kwargs)

    def set_over(self, *args, **kwargs):
        self.base_cmap.set_over(*args, **kwargs)

    def is_gray(self):
        return False

def _vis_axes():
    grid = GridSpec(10, 4,
                    left=0.02,
                    right=0.98,
                    bottom=0.02,
                    width_ratios=[1] * 4,
                    height_ratios=[1] * 10)
    axes = {'cmap': grid[0, 0],
            'deltas': grid[1:4, 0],

            'cmap-greyscale': grid[0, 1],
            'lightness-deltas': grid[1:4, 1],

            'deuteranomaly': grid[4, 0],
            'deuteranopia': grid[5, 0],
            'protanomaly': grid[4, 1],
            'protanopia': grid[5, 1],

            # 'lightness': grid[4:6, 1],
            # 'colourfulness': grid[4:6, 2],
            # 'hue': grid[4:6, 3],

            'image0': grid[0:2, 2],
            'image0-cb': grid[0:2, 3],
            'image1': grid[2:5, 2],
            'image1-cb': grid[2:5, 3],
            'image2': grid[5:8, 2],
            'image2-cb': grid[5:8, 3],
            'image3' : grid[8:,2],
            'image3-cb' : grid[8:,3]
    }

    axes = dict([(key, plt.subplot(value)) for (key, value) in axes.items()])
    axes['gamut'] = plt.subplot(grid[6:, :2], projection='3d')
    axes['gamut-toggle'] = plt.axes([0.01, 0.01, 0.08, 0.025])

    return axes


# N=256 matches the default quantization for LinearSegmentedColormap, which
# reduces quantization/aliasing artifacts (esp. in the perceptual deltas
# plot).
class viscm(object):
    def __init__(self, cm, uniform_space="CAM02-UCS",
                 name=None, N=256, N_dots=50, show_gamut=False):
        if isinstance(cm, str):
            cm = plt.get_cmap(cm)
        if name is None:
            name = cm.name

        self._sRGB1_to_uniform = cspace_converter("sRGB1", uniform_space)

        self.fig = plt.figure()
        self.fig.suptitle("Colormap evaluation: %s" % (name,), fontsize=24)
        axes = _vis_axes()

        x = np.linspace(0, 1, N)
        x_dots = np.linspace(0, 1, N_dots)
        RGB = cm(x)[:, :3]
        RGB_dots = cm(x_dots)[:, :3]

        ax = axes['cmap']
        _show_cmap(ax, RGB)
        ax.set_title("The colormap in its glory")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        def label(ax, s):
            ax.text(0.95, 0.05, s,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    transform=ax.transAxes)
        def title(ax, s):
            ax.text(0.98, 0.98, s,
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes)

        Jpapbp = self._sRGB1_to_uniform(RGB)

        def delta_ymax(values):
            return max(np.max(values) * 1.1, 0)

        ax = axes['deltas']
        local_deltas = N * np.sqrt(np.sum((Jpapbp[:-1, :] - Jpapbp[1:, :]) ** 2, axis=-1))
        #print("perceptual delta peak-to-peak: %0.2f" % (np.ptp(local_deltas),))
        ax.plot(x[1:], local_deltas)
        arclength = np.sum(local_deltas) / N
        rmse = np.std(local_deltas)
        title(ax, "Perceptual deltas")
        label(ax,
              "Length: %0.1f\nRMS deviation from flat: %0.1f (%0.1f%%)"
              % (arclength, rmse, 100 * rmse / arclength))
        ax.set_ylim(-delta_ymax(-local_deltas), delta_ymax(local_deltas))
        ax.get_xaxis().set_visible(False)

        ax = axes['cmap-greyscale']
        _show_cmap(ax, to_greyscale(RGB))
        ax.set_title("Black-and-white printed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = axes['lightness-deltas']
        ax.axhline(0, linestyle="--", color="grey")
        lightness_deltas = N * np.diff(Jpapbp[:, 0])
        ax.plot(x[1:], lightness_deltas)
        title(ax,
              "Perceptual lightness deltas")
        lightness_arclength = np.sum(np.abs(lightness_deltas)) / N
        lightness_rmse = np.std(lightness_deltas)
        label(ax,
              "Length: %0.1f\nRMS deviation from flat: %0.1f (%0.1f%%)"
              % (lightness_arclength,
                 lightness_rmse,
                 100 * lightness_rmse / np.mean(lightness_deltas)))
        #ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_ylim(-delta_ymax(-lightness_deltas), delta_ymax(lightness_deltas))
        ax.get_xaxis().set_visible(False)

        # ax = axes['lightness']
        # ax.plot(x, ciecam02.J)
        # label(ax, "Lightness (J)")
        # ax.set_ylim(0, 105)

        # ax = axes['colourfulness']
        # ax.plot(x, ciecam02.M)
        # label(ax, "Colourfulness (M)")

        # ax = axes['hue']
        # ax.plot(x, ciecam02.h)
        # label(ax, "Hue angle (h)")
        # ax.set_ylim(0, 360)

        def anom(ax, converter, name):
            _show_cmap(ax, np.clip(converter(RGB), 0, 1))
            label(ax, name)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        anom(axes['deuteranomaly'],
             _deuter50_to_sRGB1,
             "Moderate deuteranomaly")
        anom(axes['deuteranopia'],
             _deuter100_to_sRGB1,
             "Complete deuteranopia")

        anom(axes['protanomaly'],
             _prot50_to_sRGB1,
             "Moderate protanomaly")
        anom(axes['protanopia'],
             _prot100_to_sRGB1,
             "Complete protanopia")

        ax = axes['gamut']
        ax.plot(Jpapbp[:, 1], Jpapbp[:, 2], Jpapbp[:, 0])
        Jpapbp_dots = self._sRGB1_to_uniform(RGB_dots)
        ax.scatter(Jpapbp_dots[:, 1],
                   Jpapbp_dots[:, 2],
                   Jpapbp_dots[:, 0],
                   c=RGB_dots[:, :],
                   s=80)

        # Draw a wireframe indicating the sRGB gamut
        self.gamut_patch = sRGB_gamut_patch(uniform_space)
        # That function returns a patch where each face is colored to match
        # the represented colors. For present purposes we want something
        # less... colorful.
        self.gamut_patch.set_facecolor([0.5, 0.5, 0.5, 0.1])
        self.gamut_patch.set_edgecolor([0.2, 0.2, 0.2, 0.1])
        ax.add_collection3d(self.gamut_patch)
        self.gamut_patch.set_visible(show_gamut)
        ax.view_init(elev=75, azim=-75)

        self.gamut_patch_toggle = Button(axes['gamut-toggle'], "Toggle gamut")
        def toggle(*args):
            self.gamut_patch.set_visible(not self.gamut_patch.get_visible())
            plt.draw()
        self.gamut_patch_toggle.on_clicked(toggle)

        _setup_Jpapbp_axis(ax)

        images = []
        image_args = []
        example_dir = os.path.join(os.path.dirname(__file__), "examples")

        images.append(np.loadtxt(os.path.join(example_dir, "hist2d.txt")))
        image_args.append({"aspect": "equal",
                           "origin": "lower",
                           "interpolation": "nearest",
                           "vmin": 0})

        images.append(np.loadtxt(os.path.join(example_dir,
                                 "st-helens_before-modified.txt.gz")).T)
        image_args.append({})

        # Adapted from http://matplotlib.org/mpl_examples/images_contours_and_fields/pcolormesh_levels.py
        dx = dy = 0.05
        y, x = np.mgrid[-5 : 5 + dy : dy, -5 : 10 + dx : dx]
        z = np.sin(x) ** 10 + np.cos(10 + y * x) + np.cos(x) + 0.2 * y + 0.1 * x
        images.append(z)
        image_args.append({})

        # Peter Kovesi's colormap test image at http://peterkovesi.com/projects/colourmaps/colourmaptest.asc
        images.append(np.loadtxt(os.path.join(example_dir,"colourmaptest.txt")))
        image_args.append({})

        def _deuter_transform(RGBA):
            # clipping, alpha handling
            RGB = RGBA[..., :3]
            RGB = np.clip(_deuter50_to_sRGB1(RGB), 0, 1)
            return np.concatenate((RGB, RGBA[..., 3:]), axis=-1)
        deuter_cm = TransformedCMap(_deuter_transform, cm)

        for i, (image, args) in enumerate(zip(images, image_args)):
            ax = axes['image%i' % (i,)]
            ax.imshow(image, cmap=cm, **args)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax_cb = axes['image%i-cb' % (i,)]
            ax_cb.imshow(image, cmap=deuter_cm, **args)
            ax_cb.get_xaxis().set_visible(False)
            ax_cb.get_yaxis().set_visible(False)

        axes['image0'].set_title("Sample images")
        axes['image0-cb'].set_title("Moderate deuter.")

def sRGB_gamut_patch(uniform_space, resolution=20):
    step = 1.0 / resolution
    sRGB_quads = []
    sRGB_values = []
    # each entry in 'quads' is a 4x3 array where each row contains the
    # coordinates of a corner point
    for fixed in 0, 1:
        for i in range(resolution):
            for j in range(resolution):
                # R quad
                sRGB_quads.append([[fixed, i * step, j * step],
                                   [fixed, (i+1) * step, j * step],
                                   [fixed, (i+1) * step, (j+1) * step],
                                   [fixed, i * step, (j+1) * step]])
                sRGB_values.append((fixed, (i + 0.5) * step, (j + 0.5) * step,
                                    1))
                # G quad
                sRGB_quads.append([[i * step, fixed, j * step],
                                   [(i+1) * step, fixed, j * step],
                                   [(i+1) * step, fixed, (j+1) * step],
                                   [i * step, fixed, (j+1) * step]])
                sRGB_values.append(((i + 0.5) * step, fixed, (j + 0.5) * step,
                                    1))
                # B quad
                sRGB_quads.append([[i * step, j * step, fixed],
                                   [(i+1) * step, j * step, fixed],
                                   [(i+1) * step, (j+1) * step, fixed],
                                   [i * step, (j+1) * step, fixed]])
                sRGB_values.append(((i + 0.5) * step, (j + 0.5) * step, fixed,
                                    1))
    sRGB_quads = np.asarray(sRGB_quads)
    # work around colorspace transform bugginess in handling high-dim
    # arrays
    sRGB_quads_2d = sRGB_quads.reshape((-1, 3))
    Jpapbp_quads_2d = cspace_convert(sRGB_quads_2d, "sRGB1", uniform_space)
    Jpapbp_quads = Jpapbp_quads_2d.reshape((-1, 4, 3))
    gamut_patch = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
        Jpapbp_quads[:, :, [1, 2, 0]])
    gamut_patch.set_facecolor(sRGB_values)
    gamut_patch.set_edgecolor(sRGB_values)
    return gamut_patch


def sRGB_gamut_Jp_slice(Jp, uniform_space,
                        ap_lim=(-50, 50), bp_lim=(-50, 50), resolution=200):
    bp_grid, ap_grid = np.mgrid[bp_lim[0] : bp_lim[1] : resolution * 1j,
                                ap_lim[0] : ap_lim[1] : resolution * 1j]
    Jp_grid = Jp * np.ones((resolution, resolution))
    Jpapbp = np.concatenate((Jp_grid[:, :, np.newaxis],
                             ap_grid[:, :, np.newaxis],
                             bp_grid[:, :, np.newaxis]),
                            axis=2)
    sRGB = cspace_convert(Jpapbp, uniform_space, "sRGB1")
    sRGBA = np.concatenate((sRGB, np.ones(sRGB.shape[:2] + (1,))),
                           axis=2)
    sRGBA[np.any((sRGB < 0) | (sRGB > 1), axis=-1)] = [0, 0, 0, 0]
    return sRGBA


def draw_pure_hue_angles(ax):
    # Pure hue angles from CIECAM-02
    for color, angle in [("r", 20.14),
                         ("y", 90.00),
                         ("g", 164.25),
                         ("b", 237.53)]:
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        ax.plot([0, x * 1000], [0, y * 1000], color + "--")


def draw_sRGB_gamut_Jp_slice(ax, Jp, uniform_space,
                             ap_lim=(-50, 50), bp_lim=(-50, 50),
                             **kwargs):
    sRGB = sRGB_gamut_Jp_slice(Jp, uniform_space,
                               ap_lim=ap_lim, bp_lim=bp_lim, **kwargs)
    im = ax.imshow(sRGB, aspect="equal",
                   extent=ap_lim + bp_lim, origin="lower")
    draw_pure_hue_angles(ax)
    ax.set_xlim(ap_lim)
    ax.set_ylim(bp_lim)
    return im

# def sRGB_gamut_J_slice(J,
#                        ap_lim=(-50, 50), bp_lim=(-50, 50), resolution=200):
#     a_grid, b_grid = np.mgrid[ap_lim[0] : ap_lim[1] : resolution * 1j,
#                               bp_lim[0] : bp_lim[1] : resolution * 1j]
#     J_grid = J * np.ones((resolution, resolution))
#     h = np.rad2deg(np.arctan2(b_grid, a_grid))
#     M = np.hypot(a_grid, b_grid)
#     XYZ = ViewingConditions.sRGB.CIECAM02_to_XYZ(J=J_grid, M=M, h=h)
#     sRGB = XYZ_to_sRGB(XYZ)
#     sRGB[np.any((sRGB < 0) | (sRGB > 1), axis=-1)] = np.nan
#     return sRGB


def _viscm_editor_axes():
    grid = GridSpec(1, 2,
                    width_ratios=[5, 1],
                    height_ratios=[6, 1])
    axes = {'bezier': grid[0, 0],
            'cm': grid[0, 1]}

    axes = dict([(key, plt.subplot(value)) for (key, value) in axes.items()])
    return axes


class viscm_editor(object):
    def __init__(self, uniform_space="CAM02-UCS",
                 min_Jp=15, max_Jp=95, xp=None, yp=None):
        from .bezierbuilder import BezierModel, BezierBuilder

        self._uniform_space = uniform_space

        axes = _viscm_editor_axes()

        ax_btn_wireframe = plt.axes([0.7, 0.15, 0.1, 0.025])
        self.btn_wireframe = Button(ax_btn_wireframe, 'Show 3D gamut')
        self.btn_wireframe.on_clicked(self.plot_3d_gamut)

        ax_btn_wireframe = plt.axes([0.81, 0.15, 0.1, 0.025])
        self.btn_save = Button(ax_btn_wireframe, 'Save colormap')
        self.btn_save.on_clicked(self.save_colormap)

        ax_btn_props = plt.axes([0.81, 0.1, 0.1, 0.025])
        self.btn_props = Button(ax_btn_props, 'Properties')
        self.btn_props.on_clicked(self.show_viscm)
        self.prop_windows = []

        axcolor = 'None'
        ax_jp_min = plt.axes([0.1, 0.1, 0.5, 0.03], axisbg=axcolor)
        ax_jp_min.imshow(np.linspace(0, 100, 101).reshape(1, -1), cmap='gray')
        ax_jp_min.set_xlim(0, 100)

        ax_jp_max = plt.axes([0.1, 0.15, 0.5, 0.03], axisbg=axcolor)
        ax_jp_max.imshow(np.linspace(0, 100, 101).reshape(1, -1), cmap='gray')

        self.jp_min_slider = Slider(ax_jp_min, r"$J'_\mathrm{min}$", 0, 100, valinit=min_Jp)
        self.jp_max_slider = Slider(ax_jp_max, r"$J'_\mathrm{max}$", 0, 100, valinit=max_Jp)

        self.jp_min_slider.on_changed(self._jp_update)
        self.jp_max_slider.on_changed(self._jp_update)

        if xp is None:
            xp = [-2.0591553836234482, 59.377014829142524,
                  43.552546744036135, 4.7670857511283202,
                  -9.5059638942617539]

        if yp is None:
            yp = [-25.664893617021221, -21.941489361702082,
                  38.874113475177353, 20.567375886524871,
                  32.047872340425585]

        self.bezier_model = BezierModel(xp, yp)
        self.cmap_model = BezierCMapModel(self.bezier_model,
                                          self.jp_min_slider.val,
                                          self.jp_max_slider.val,
                                          uniform_space)
        self.highlight_point_model = HighlightPointModel(self.cmap_model, 0.5)

        self.bezier_builder = BezierBuilder(axes['bezier'], self.bezier_model)
        self.bezier_gamut_viewer = GamutViewer2D(axes['bezier'],
                                                 self.highlight_point_model,
                                                 uniform_space)
        tmp = HighlightPoint2DView(axes['bezier'],
                                   self.highlight_point_model)
        self.bezier_highlight_point_view = tmp

        #draw_pure_hue_angles(axes['bezier'])
        axes['bezier'].set_xlim(-100, 100)
        axes['bezier'].set_ylim(-100, 100)

        self.cmap_view = CMapView(axes['cm'], self.cmap_model)
        self.cmap_highlighter = HighlightPointBuilder(
            axes['cm'],
            self.highlight_point_model)

        print("Click sliders at bottom to change min/max lightness")
        print("Click on colorbar to adjust gamut view")
        print("Click-drag to move control points, ")
        print("  shift-click to add, control-click to delete")

    def plot_3d_gamut(self, event):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        self.wireframe_view = WireframeView(ax,
                                            self.cmap_model,
                                            self.highlight_point_model,
                                            self._uniform_space)
        plt.show()

    def save_colormap(self, event):
        import textwrap

        template = textwrap.dedent('''
        from matplotlib.colors import ListedColormap
        from numpy import nan, inf

        # Used to reconstruct the colormap in viscm
        parameters = {{'xp': {xp},
                      'yp': {yp},
                      'min_Jp': {min_Jp},
                      'max_Jp': {max_Jp}}}

        cm_data = {array_list}

        test_cm = ListedColormap(cm_data, name=__file__)


        if __name__ == "__main__":
            import matplotlib.pyplot as plt
            import numpy as np

            try:
                from viscm import viscm
                viscm(test_cm)
            except ImportError:
                print("viscm not found, falling back on simple display")
                plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',
                           cmap=test_cm)
            plt.show()
        ''')

        rgb, _ = self.cmap_model.get_sRGB(num=256)
        with open('/tmp/new_cm.py', 'w') as f:
            array_list = np.array2string(rgb, max_line_width=78,
                                         prefix='cm_data = ',
                                         separator=',')

            xp, yp = self.cmap_model.bezier_model.get_control_points()

            data = dict(array_list=array_list,
                        xp=xp,
                        yp=yp,
                        min_Jp=self.cmap_model.min_Jp,
                        max_Jp=self.cmap_model.max_Jp)

            f.write(template.format(**data))

            print("*" * 50)
            print("Saved colormap to /tmp/new_cm.py")
            print("*" * 50)

    def show_viscm(self, event):
        cm = LinearSegmentedColormap.from_list(
            'test_cm',
            self.cmap_model.get_sRGB(num=256)[0])
        self.prop_windows.append(viscm(cm, name='test_cm'))
        plt.show()

    def _jp_update(self, val):
        jp_min = self.jp_min_slider.val
        jp_max = self.jp_max_slider.val

        smallest, largest = min(jp_min, jp_max), max(jp_min, jp_max)
        if (jp_min > smallest) or (jp_max < largest):
            self.jp_min_slider.set_val(smallest)
            self.jp_max_slider.set_val(largest)

        self.cmap_model.set_Jp_minmax(smallest, largest)

class BezierCMapModel(object):
    def __init__(self, bezier_model, min_Jp, max_Jp, uniform_space):
        self.bezier_model = bezier_model
        self.min_Jp = min_Jp
        self.max_Jp = max_Jp
        self.trigger = Trigger()
        self.uniform_to_sRGB1 = cspace_converter(uniform_space, "sRGB1")

        self.bezier_model.trigger.add_callback(self.trigger.fire)

    def set_Jp_minmax(self, min_Jp, max_Jp):
        self.min_Jp = min_Jp
        self.max_Jp = max_Jp
        self.trigger.fire()

    def get_Jpapbp_at(self, at):
        ap, bp = self.bezier_model.get_bezier_points_at(at)
        Jp = (self.max_Jp - self.min_Jp) * at + self.min_Jp
        return Jp, ap, bp

    def get_Jpapbp(self, num=200):
        return self.get_Jpapbp_at(np.linspace(0, 1, num))

    def get_sRGB(self, num=200):
        # Return sRGB and out-of-gamut mask
        Jp, ap, bp = self.get_Jpapbp(num=num)
        sRGB = self.uniform_to_sRGB1(np.column_stack((Jp, ap, bp)))
        oog = np.any((sRGB > 1) | (sRGB < 0), axis=-1)
        sRGB[oog, :] = np.nan
        return sRGB, oog


class CMapView(object):
    def __init__(self, ax, cmap_model):
        self.ax = ax
        self.cmap_model = cmap_model

        rgb_display, oog_display = self._drawable_arrays()
        self.image = self.ax.imshow(rgb_display, extent=(0, 0.2, 0, 1),
                                    origin="lower")
        self.gamut_alert_image = self.ax.imshow(oog_display,
                                                extent=(0.05, 0.15, 0, 1),
                                                origin="lower")
        self.ax.set_xlim(0, 0.2)
        self.ax.set_ylim(0, 1)
        self.ax.get_xaxis().set_visible(False)

        self.cmap_model.trigger.add_callback(self._refresh)

    def _drawable_arrays(self):
        rgb, oog = self.cmap_model.get_sRGB()
        rgb_display = rgb[:, np.newaxis, :]
        oog_display = np.empty((rgb.shape[0], 1, 4))
        oog_display[...] = [0, 0, 0, 0]
        oog_display[oog, :, :] = [0, 1, 1, 1]
        return rgb_display, oog_display

    def _refresh(self):
        rgb_display, oog_display = self._drawable_arrays()
        self.image.set_data(rgb_display)
        self.gamut_alert_image.set_data(oog_display)


class HighlightPointModel(object):
    def __init__(self, cmap_model, point):
        self._cmap_model = cmap_model
        self._point = point
        self.trigger = Trigger()

        self._cmap_model.trigger.add_callback(self.trigger.fire)

    def get_point(self):
        return self._point

    def set_point(self, point):
        self._point = point
        self.trigger.fire()

    def get_Jpapbp(self):
        return self._cmap_model.get_Jpapbp_at(self._point)


class HighlightPointBuilder(object):
    def __init__(self, ax, highlight_point_model):
        self.ax = ax
        self.highlight_point_model = highlight_point_model

        self.canvas = self.ax.figure.canvas
        self._in_drag = False
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event",
                                self._on_button_release)

        self.marker_line = self.ax.axhline(highlight_point_model.get_point(),
                                           linewidth=3, color="r")

        self.highlight_point_model.trigger.add_callback(self._refresh)

    def _on_button_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self._in_drag = True
        self.highlight_point_model.set_point(event.ydata)

    def _on_motion(self, event):
        if self._in_drag and event.ydata is not None:
            self.highlight_point_model.set_point(event.ydata)

    def _on_button_release(self, event):
        if event.button != 1:
            return
        self._in_drag = False

    def _refresh(self):
        point = self.highlight_point_model.get_point()
        self.marker_line.set_data([0, 1], [point, point])
        self.canvas.draw()


class GamutViewer2D(object):
    def __init__(self, ax, highlight_point_model, uniform_space,
                 ap_lim=(-50, 50), bp_lim=(-50, 50)):
        self.ax = ax
        self.highlight_point_model = highlight_point_model
        self.ap_lim = ap_lim
        self.bp_lim = bp_lim
        self.uniform_space = uniform_space

        self.bgcolors = {"light": (0.9, 0.9, 0.9),
                         "dark": (0.1, 0.1, 0.1)}
        # We want some hysteresis, so that there's no point where wiggling the
        # line back and forth causes background flickering.
        self.bgcolor_ranges = {"light": (0, 60), "dark": (40, 100)}
        self.bg_opposites = {"light": "dark", "dark": "light"}
        self.bg = "light"
        self.ax.set_axis_bgcolor(self.bgcolors[self.bg])

        self.image = self.ax.imshow([[[0, 0, 0]]], aspect="equal",
                                    extent=ap_lim + bp_lim,
                                    origin="lower")

        self.highlight_point_model.trigger.add_callback(self._refresh)

    def _refresh(self):
        Jp, _, _ = self.highlight_point_model.get_Jpapbp()
        low, high = self.bgcolor_ranges[self.bg]
        if not (low <= Jp <= high):
            self.bg = self.bg_opposites[self.bg]
            self.ax.set_axis_bgcolor(self.bgcolors[self.bg])
        sRGB = sRGB_gamut_Jp_slice(Jp, self.uniform_space,
                                   self.ap_lim, self.bp_lim)
        self.image.set_data(sRGB)


class HighlightPoint2DView(object):
    def __init__(self, ax, highlight_point_model):
        self.ax = ax
        self.highlight_point_model = highlight_point_model

        _, ap, bp = self.highlight_point_model.get_Jpapbp()
        self.marker = self.ax.plot([ap], [bp], "y.", mew=3)[0]

        self.highlight_point_model.trigger.add_callback(self._refresh)

    def _refresh(self):
        _, ap, bp = self.highlight_point_model.get_Jpapbp()
        self.marker.set_data([ap], [bp])
        self.ax.figure.canvas.draw()


class WireframeView(object):
    def __init__(self, ax, cmap_model, highlight_point_model, uniform_space):
        self.ax = ax
        self.cmap_model = cmap_model
        self.highlight_point_model = highlight_point_model

        Jp, ap, bp = self.cmap_model.get_Jpapbp()
        self.line = self.ax.plot([0, 10], [0, 10])[0]
        #self.line = self.ax.plot(Jp, ap, bp)[0]

        Jp, ap, bp = self.highlight_point_model.get_Jpapbp()
        self.marker = self.ax.plot([Jp], [ap], [bp], "y.", mew=3)[0]

        gamut_patch = sRGB_gamut_patch(uniform_space)
        # That function returns a patch where each face is colored to match
        # the represented colors. For present purposes we want something
        # less... colorful.
        gamut_patch.set_facecolor([0.5, 0.5, 0.5, 0.1])
        gamut_patch.set_edgecolor([0.2, 0.2, 0.2, 0.1])
        self.ax.add_collection3d(gamut_patch)

        _setup_Jpapbp_axis(self.ax)

        #self.cmap_model.trigger.add_callback(self._refresh_line)
        #self.highlight_point_model.trigger.add_callback(self._refresh_point)
        self._refresh_line()
        self._refresh_point()

    def _refresh_line(self):
        Jp, ap, bp = self.cmap_model.get_Jpapbp()
        self.line.set_data(ap, bp)
        self.line.set_3d_properties(zs=Jp)
        self.ax.figure.canvas.draw()

    def _refresh_point(self):
        Jp, ap, bp = self.highlight_point_model.get_Jpapbp()
        self.marker.set_data([ap], [bp])
        self.marker.set_3d_properties(zs=[Jp])
        self.ax.figure.canvas.draw()


def main(argv):
    import argparse

    # Usage:
    #   python -m viscm
    #   python -m viscm edit
    #   python -m viscm edit <file.py>
    #      (file.py must define some appropriate globals)
    #   python -m viscm view <file.py>
    #      (file.py must define a global named "test_cm")
    #   python -m viscm view "matplotlib builtin colormap"
    #   python -m viscm view --save=foo.png ...

    parser = argparse.ArgumentParser(
        prog="python -m viscm",
        description="A colormap tool.",
    )
    parser.add_argument("action", metavar="ACTION",
                        help="'edit' or 'view'",
                        choices=["edit", "view", "show"],
                        default="edit",
                        nargs="?")
    parser.add_argument("colormap", metavar="COLORMAP",
                        default=None,
                        help="A .py file saved from the editor, or "
                             "the name of a matplotlib builtin colormap",
                        nargs="?")
    parser.add_argument("--uniform-space", metavar="SPACE",
                        default="CAM02-UCS",
                        dest="uniform_space",
                        help="The perceptually uniform space to use. Usually "
                        "you should leave this alone. You can pass 'CIELab' "
                        "if you're curious how uniform some colormap is in "
                        "CIELab space. You can pass 'buggy-CAM02-UCS' if "
                        "you're trying to reproduce the matplotlib colormaps "
                        "(which turn out to have had a small bug in the "
                        "assumed sRGB viewing conditions) from their bezier "
                        "curves.")
    parser.add_argument("--save", metavar="FILE",
                        default=None,
                        help="Immediately save visualization to a file (view-mode only).")
    parser.add_argument("--quit", default=False, action="store_true",
                        help="Quit immediately after starting (useful with --save).")
    args = parser.parse_args(argv)

    params = {}
    cmap = None
    if args.colormap:
        if os.path.isfile(args.colormap):
            ns = {'__name__': '',
                  '__file__': os.path.basename(args.colormap),
            }

            with open(args.colormap) as f:
                code = compile(f.read(),
                               os.path.basename(args.colormap),
                               'exec')
                exec(code, globals(), ns)

            params = ns.get('parameters', {})
            if "min_JK" in params:
                params["min_Jp"] = params.pop("min_JK")
                params["max_Jp"] = params.pop("max_JK")
            cmap = ns.get("test_cm", None)
        else:
            cmap = plt.get_cmap(args.colormap)

    uniform_space = args.uniform_space
    if uniform_space == "buggy-CAM02-UCS":
        uniform_space = buggy_CAM02UCS
    # Easter egg! I keep typing 'show' instead of 'view' so accept both
    if args.action in ("view", "show"):
        if cmap is None:
            sys.exit("Please specify a colormap")
        v = viscm(cmap, uniform_space=uniform_space)
        if args.save is not None:
            v.fig.set_size_inches(20, 12)
            v.fig.savefig(args.save)
    elif args.action == "edit":
        if params is None:
            sys.exit("Sorry, I don't know how to edit the specified colormap")
        # Hold a reference so it doesn't get GC'ed
        v = viscm_editor(uniform_space=uniform_space, **params)
    else:
        raise RuntimeError("can't happen")

    if args.quit:
        sys.exit()

    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])

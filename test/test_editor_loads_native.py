import json

import pytest

from viscm.gui import Colormap, viscm_editor


def approxeq(x, y, *, err=0.0001):
    return abs(y - x) < err


@pytest.mark.parametrize(
    "colormap_file",
    [
        "viscm/examples/sample_linear.jscm",
        "viscm/examples/sample_diverging.jscm",
        "viscm/examples/sample_diverging_continuous.jscm",
    ],
)
def test_editor_loads_native(colormap_file):
    with open(colormap_file) as f:
        data = json.loads(f.read())
    cm = Colormap(None, "CatmulClark", "CAM02-UCS")
    cm.load(colormap_file)
    viscm = viscm_editor(
        uniform_space=cm.uniform_space,
        cmtype=cm.cmtype,
        method=cm.method,
        **cm.params,
    )
    assert viscm.name == data["name"]

    extensions = data["extensions"]["https://matplotlib.org/viscm"]
    xp, yp, fixed = viscm.control_point_model.get_control_points()

    assert extensions["fixed"] == fixed
    assert len(extensions["xp"]) == len(xp)
    assert len(extensions["yp"]) == len(yp)
    assert len(xp) == len(yp)
    for i in range(len(xp)):
        assert extensions["xp"][i] == xp[i]
        assert extensions["yp"][i] == yp[i]
    assert extensions["min_Jp"] == viscm.min_Jp
    assert extensions["max_Jp"] == viscm.max_Jp
    assert extensions["filter_k"] == viscm.cmap_model.filter_k
    assert extensions["cmtype"] == viscm.cmtype

    # Decode hexadecimal-encoded colormap string (grouped in units of 3 pairs of
    # two-character (0-255) values) to 3-tuples of floats (0-1).
    colors_hex = data["colors"]
    colors_hex = [colors_hex[i : i + 6] for i in range(0, len(colors_hex), 6)]
    colors = [
        [int(c[i : i + 2], 16) / 255 for i in range(0, len(c), 2)] for c in colors_hex
    ]

    editor_colors = viscm.cmap_model.get_sRGB(num=256)[0].tolist()

    for i in range(len(colors)):
        for z in range(3):
            assert approxeq(colors[i][z], editor_colors[i][z], err=0.01)


# import matplotlib as mpl
# try:
#     from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# except ImportError:
#     try:
#         from matplotlib.backends.backend_qt5agg import (
#             FigureCanvasQTAgg as FigureCanvas
#         )
#     except ImportError:
#         from matplotlib.backends.backend_qt4agg import (
#             FigureCanvasQTAgg as FigureCanvas
#         )
# from matplotlib.backends.qt_compat import QtCore, QtGui
#
# def test_editor_add_point():
#     # Testing linear
#
#     fig = plt.figure()
#     figure_canvas = FigureCanvas(fig)
#     linear = viscm_editor(
#         min_Jp=40,
#         max_Jp=60,
#         xp=[-10, 10],
#         yp=[0,0],
#         figure=fig,
#         cmtype="linear",
#     )
#
#     Jp, ap, bp = linear.cmap_model.get_Jpapbp(3)
#     eJp, eap, ebp = [40, 50, 60], [-10, 0, 10], [0, 0, 0]
#     for i in range(3):
#         assert approxeq(Jp[i], eJp[i])
#         assert approxeq(ap[i], eap[i])
#         assert approxeq(bp[i], ebp[i])
#     rgb = linear.cmap_model.get_sRGB(3)[0]
#     ergb = [[ 0.27446483,  0.37479529,  0.34722738],
#             [ 0.44884374,  0.44012037,  0.43848162],
#             [ 0.63153956,  0.49733664,  0.53352363]]
#     for i in range(3):
#         for z in range(3):
#             assert approxeq(rgb[i][z], ergb[i][z])


#     # Testing adding a point to linear
#     linear.bezier_builder.mode = "add"
#     qtEvent = QtGui.QMouseEvent(
#         QtCore.QEvent.MouseButtonPress,
#         QtCore.QPoint(),
#         QtCore.Qt.LeftButton,
#         QtCore.Qt.LeftButton,
#         QtCore.Qt.ShiftModifier,
#     )
#     event = mpl.backend_bases.MouseEvent(
#         "button_press_event",
#         figure_canvas,
#         0,
#         10,
#         guiEvent=qtEvent,
#     )
#     event.xdata = 0
#     event.ydata = 10
#     event.inaxes = linear.bezier_builder.ax
#     linear.bezier_builder.on_button_press(event)
#     Jp, ap, bp = linear.cmap_model.get_Jpapbp(3)
#     eJp, eap, ebp = [40, 50, 60], [-10, 0, 10], [0, 5, 0]
#     for i in range(3):
#         assert approxeq(Jp[i], eJp[i])
#         assert approxeq(ap[i], eap[i])
#         assert approxeq(bp[i], ebp[i])
#     rgb = linear.cmap_model.get_sRGB(3)[0]
#     ergb = [[ 0.27446483,  0.37479529,  0.34722738],
#             [ 0.46101392,  0.44012069,  0.38783966],
#             [ 0.63153956,  0.49733664,  0.53352363]]
#     for i in range(3):
#         for z in range(3):
#             assert approxeq(rgb[i][z], ergb[i][z])

#     # Removing a point from linear
#     linear.bezier_builder.mode = "remove"
#     qtEvent = QtGui.QMouseEvent(
#         QtCore.QEvent.MouseButtonPress,
#         QtCore.QPoint(),
#         QtCore.Qt.LeftButton,
#         QtCore.Qt.LeftButton,
#         QtCore.Qt.ControlModifier,
#     )
#     event = mpl.backend_bases.MouseEvent(
#         "button_press_event",
#         figure_canvas,
#         0,
#         10,
#         guiEvent=qtEvent,
#     )
#     event.xdata = 0
#     event.ydata = 10
#     event.inaxes = linear.bezier_builder.ax
#     linear.bezier_builder.on_button_press(event)
#     # Jp, ap, bp = linear.cmap_model.get_Jpapbp(3)
#     # print(Jp, ap, bp)
#     # print(rgb)
#     # use mpl transformations

#     print(linear.control_point_model.get_control_points())
#     # print(linear.cmap_model.get_Jpapbp(3))

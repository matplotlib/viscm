from viscm.gui import *
from viscm.bezierbuilder import *
import numpy as np
import matplotlib as mpl
from matplotlib.backends.qt_compat import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

viscm_editors = {path: viscm_editor(cmtype=loadfile(path)[1], **(loadfile(path)[0])) for path in
        {"viscm/examples/sample_linear.jscm",
         "viscm/examples/sample_diverging.jscm",
         "viscm/examples/sample_diverging_continuous.jscm"}}
viscm_viewers = {path: viscm(loadfile(path)[4], name=loadfile(path)) for path in
        {"viscm/examples/sample_linear_foreign.jscm",
         "viscm/examples/sample_diverging_foreign.jscm",
         "viscm/examples/sample_diverging_continuous_foreign.jscm"}}

def test_editor_loads_native():
    for k, v in viscm_editors.items():
        with open(k) as f:
            data = json.loads(f.read())
        assert v.name == data["name"]

        extensions = data["extensions"]["https://matplotlib.org/viscm"]
        xp, yp, fixed = v.control_point_model.get_control_points()

        assert extensions["fixed"] == fixed
        assert len(extensions["xp"]) == len(xp)
        assert len(extensions["yp"]) == len(yp)
        assert len(xp) == len(yp)
        for i in range(len(xp)): 
            assert extensions["xp"][i] == xp[i]
            assert extensions["yp"][i] == yp[i]
        assert extensions["min_Jp"] == v.min_Jp
        assert extensions["max_Jp"] == v.max_Jp
        assert extensions["filter_k"] == v.filter_k
        assert extensions["cmtype"] == v.cmtype

        colors = data["colors"]
        colors = [[int(c[i:i + 2], 16) / 256 for i in range(0, 6, 2)] for c in [colors[i:i + 6] for i in range(0, len(colors), 6)]]
        editor_colors = v.cmap_model.get_sRGB(num=256)[0].tolist()
        for i in range(len(colors)):
            for z in range(3):
                assert colors[i][z] == np.rint(editor_colors[i][z] / 256)

def test_editor_add_point():
    # Testing linear

    fig = plt.figure()
    figure_canvas = FigureCanvas(fig)
    linear = viscm_editor(min_Jp=40, max_Jp=60, xp=[-10, 10], yp=[0,0], figure=fig, cmtype="linear")

    Jp, ap, bp = linear.cmap_model.get_Jpapbp(3)
    eJp, eap, ebp = [40, 50, 60], [-10, 0, 10], [0, 0, 0]
    for i in range(3):
        assert approxeq(Jp[i], eJp[i])
        assert approxeq(ap[i], eap[i])
        assert approxeq(bp[i], ebp[i])
    rgb = linear.cmap_model.get_sRGB(3)[0]
    ergb = [[ 0.27446483,  0.37479529,  0.34722738],
            [ 0.44884374,  0.44012037,  0.43848162],
            [ 0.63153956,  0.49733664,  0.53352363]]
    for i in range(3):
        for z in range(3):
            assert approxeq(rgb[i][z], ergb[i][z])
    

    # Testing adding a point to linear
    linear.bezier_builder.mode = "add"
    qtEvent = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, QtCore.QPoint(), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier)
    event = mpl.backend_bases.MouseEvent("button_press_event", figure_canvas, 0, 10, guiEvent=qtEvent)
    event.xdata = 0
    event.ydata = 10
    event.inaxes = linear.bezier_builder.ax
    linear.bezier_builder.on_button_press(event)
    Jp, ap, bp = linear.cmap_model.get_Jpapbp(3)
    eJp, eap, ebp = [40, 50, 60], [-10, 0, 10], [0, 5, 0]
    for i in range(3):
        assert approxeq(Jp[i], eJp[i])
        assert approxeq(ap[i], eap[i])
        assert approxeq(bp[i], ebp[i])
    rgb = linear.cmap_model.get_sRGB(3)[0]
    ergb = [[ 0.27446483,  0.37479529,  0.34722738],
            [ 0.46101392,  0.44012069,  0.38783966],
            [ 0.63153956,  0.49733664,  0.53352363]]
    for i in range(3):
        for z in range(3):
            assert approxeq(rgb[i][z], ergb[i][z])

    # Removing a point from linear

    # Jp, ap, bp = linear.cmap_model.get_Jpapbp(3)
    # print(Jp, ap, bp)
    # print(rgb)

    # print(linear.control_point_model.get_control_points())
    # print(linear.cmap_model.get_Jpapbp(3))



def approxeq(x, y, err=0.0001):
    return abs(y - x) < err


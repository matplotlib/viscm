import sys

import matplotlib.pyplot as plt

from viscm import gui


def cli():
    import argparse

    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="python -m viscm",
        description="A colormap tool.",
    )
    parser.add_argument(
        "action",
        metavar="ACTION",
        help="'edit' or 'view' (or 'show', same as 'view')",
        choices=["edit", "view", "show"],
        default="edit",
        nargs="?",
    )
    parser.add_argument(
        "colormap",
        metavar="COLORMAP",
        default=None,
        help="A .json file saved from the editor, a .py file containing"
        " a global named `test_cm`, or the name of a matplotlib builtin"
        " colormap",
        nargs="?",
    )
    parser.add_argument(
        "--uniform-space",
        metavar="SPACE",
        default="CAM02-UCS",
        dest="uniform_space",
        help="The perceptually uniform space to use. Usually "
        "you should leave this alone. You can pass 'CIELab' "
        "if you're curious how uniform some colormap is in "
        "CIELab space. You can pass 'buggy-CAM02-UCS' if "
        "you're trying to reproduce the matplotlib colormaps "
        "(which turn out to have had a small bug in the "
        "assumed sRGB viewing conditions) from their bezier "
        "curves.",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="linear",
        choices=["linear", "diverging", "diverging-continuous"],
        help=(
            "Choose a colormap type. Supported options are 'linear', 'diverging',"
            " and 'diverging-continuous"
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="CatmulClark",
        choices=["Bezier", "CatmulClark"],
        help=(
            "Choose a spline construction method. 'CatmulClark' is the default, but"
            " you may choose the legacy option 'Bezier'"
        ),
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        default=None,
        help="Immediately save visualization to a file " "(view-mode only).",
    )
    parser.add_argument(
        "--quit",
        default=False,
        action="store_true",
        help="Quit immediately after starting " "(useful with --save).",
    )
    args = parser.parse_args(argv)

    cm = gui.Colormap(args.type, args.method, args.uniform_space)
    app = gui.QtWidgets.QApplication([])

    if args.colormap:
        cm.load(args.colormap)

    # Easter egg! I keep typing 'show' instead of 'view' so accept both
    if args.action in ("view", "show"):
        if cm is None:
            sys.exit("Please specify a colormap")
        fig = plt.figure()
        figure_canvas = gui.FigureCanvas(fig)
        v = gui.viscm(cm.cmap, name=cm.name, figure=fig, uniform_space=cm.uniform_space)
        mainwindow = gui.ViewerWindow(figure_canvas, v, cm.name)
        if args.save is not None:
            v.figure.set_size_inches(20, 12)
            v.figure.savefig(args.save)
    elif args.action == "edit":
        if not cm.can_edit:
            sys.exit("Sorry, I don't know how to edit the specified colormap")
        # Hold a reference so it doesn't get GC'ed
        fig = plt.figure()
        figure_canvas = gui.FigureCanvas(fig)
        v = gui.viscm_editor(
            figure=fig,
            uniform_space=cm.uniform_space,
            cmtype=cm.cmtype,
            method=cm.method,
            **cm.params,
        )
        mainwindow = gui.EditorWindow(figure_canvas, v)
    else:
        raise RuntimeError("can't happen")

    if args.quit:
        sys.exit()

    figure_canvas.setSizePolicy(
        gui.QtWidgets.QSizePolicy.Expanding, gui.QtWidgets.QSizePolicy.Expanding
    )
    figure_canvas.updateGeometry()

    mainwindow.resize(800, 600)
    mainwindow.show()

    # PyQt messes up signal handling by default. Python signal handlers (e.g.,
    # the default handler for SIGINT that raises KeyboardInterrupt) can only
    # run when we enter the Python interpreter, which doesn't happen while
    # idling in the Qt mainloop. (Unless we register a timer to poll
    # explicitly.) So here we unregister Python's default signal handler and
    # replace it with... the *operating system's* default signal handler, so
    # instead of a KeyboardInterrupt our process just exits.
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app.exec_()


if __name__ == "__main__":
    cli()

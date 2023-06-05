import sys
from pathlib import Path
from typing import Union

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
        help="Immediately save visualization to a file (view-mode only).",
    )
    parser.add_argument(
        "--quit",
        default=False,
        action="store_true",
        help="Quit immediately after starting (useful with --save).",
    )
    args = parser.parse_args(argv)

    app = gui.QtWidgets.QApplication([])

    try:
        mainwindow = _make_window(
            action=args.action,
            cmap=args.colormap,
            cmap_type=args.type,
            cmap_spline_method=args.method,
            cmap_uniform_space=args.uniform_space,
            save=Path(args.save) if args.save else None,
            quit_immediately=args.quit,
        )
    except Exception as e:
        sys.exit(str(e))

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


def _make_window(
    *,
    action: str,
    cmap: Union[str, None],
    cmap_type: str,
    cmap_spline_method: str,
    cmap_uniform_space: str,
    save: Union[Path, None],
    quit_immediately: bool,
) -> Union[gui.ViewerWindow, gui.EditorWindow]:
    # Hold a reference so it doesn't get GC'ed
    fig = plt.figure()
    figure_canvas = gui.FigureCanvas(fig)

    cm = gui.Colormap(cmap_type, cmap_spline_method, cmap_uniform_space)
    if cmap:
        cm.load(cmap)

    v: Union[gui.viscm, gui.viscm_editor]
    # Easter egg! I keep typing 'show' instead of 'view' so accept both
    if action in ("view", "show"):
        if cm is None:
            raise RuntimeError("Please specify a colormap")

        v = gui.viscm(cm.cmap, name=cm.name, figure=fig, uniform_space=cm.uniform_space)
        window = gui.ViewerWindow(figure_canvas, v, cm.name)
        if save is not None:
            v.figure.set_size_inches(20, 12)
            v.figure.savefig(str(save))
    elif action == "edit":
        if not cm.can_edit:
            sys.exit("Sorry, I don't know how to edit the specified colormap")

        v = gui.viscm_editor(
            figure=fig,
            uniform_space=cm.uniform_space,
            cmtype=cm.cmtype,
            method=cm.method,
            **cm.params,
        )
        window = gui.EditorWindow(figure_canvas, v)
    else:
        raise RuntimeError(
            "Action must be 'edit', 'view', or 'show'. This should never happen.",
        )

    if quit_immediately:
        sys.exit()

    figure_canvas.setSizePolicy(
        gui.QtWidgets.QSizePolicy.Expanding, gui.QtWidgets.QSizePolicy.Expanding
    )
    figure_canvas.updateGeometry()

    window.resize(800, 600)
    return window


if __name__ == "__main__":
    cli()

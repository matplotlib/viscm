from pathlib import Path

import pytest

from viscm.cli import _make_window


class TestGui:
    def test_gui_view_opens(self, qtbot):
        window = _make_window(
            action="view",
            cmap="viridis",
            cmap_type="linear",
            cmap_spline_method="CatmulClark",
            cmap_uniform_space="CAM02-UCS",
            save=None,
            quit_immediately=False,
        )
        window.show()
        qtbot.addWidget(window)

        assert window.isVisible()

    @pytest.mark.xfail(
        reason="Unknown. See https://github.com/matplotlib/viscm/issues/71",
    )
    def test_gui_edit_pyfile_opens(self, tests_data_dir: Path, qtbot):
        """Reproduce viridis from README instructions.

        https://github.com/matplotlib/viscm/pull/58
        """
        window = _make_window(
            action="edit",
            cmap=str(tests_data_dir / "option_d.py"),
            cmap_type="linear",
            cmap_spline_method="Bezier",
            cmap_uniform_space="buggy-CAM02-UCS",
            save=None,
            quit_immediately=False,
        )
        window.show()
        qtbot.addWidget(window)

        assert window.isVisible()

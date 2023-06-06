from pathlib import Path

import pytest
import pytest_xvfb

from viscm.cli import _make_window

xvfb_installed = pytest_xvfb.xvfb_instance is not None


@pytest.mark.skipif(
    not xvfb_installed,
    reason="Xvfb must be installed for this test.",
)
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

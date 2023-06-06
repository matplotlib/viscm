viscm
=====

This is a little tool for analyzing colormaps and creating new colormaps.

Try::

  $ pip install viscm
  $ python -m viscm view jet
  $ python -m viscm edit

There is some information available about how to interpret the
resulting visualizations and use the editor tool `on this website
<https://bids.github.io/colormap/>`_.

Downloads:
  * https://pypi.python.org/pypi/viscm/
  * https://anaconda.org/conda-forge/viscm/

Code and bug tracker:
  https://github.com/matplotlib/viscm

Contact:
  Nathaniel J. Smith <njs@pobox.com> and Stéfan van der Walt <stefanv@berkeley.edu>

Dependencies:
  * Python 3.8+
  * `colorspacious <https://pypi.python.org/pypi/colorspacious>`_
  * Matplotlib
  * NumPy

License:
  MIT, see `LICENSE <LICENSE>`__ for details.

Reproducing viridis
-------------------

Load [viridis AKA option_d.py](https://github.com/BIDS/colormap/) using:

```
python -m viscm --uniform-space buggy-CAM02-UCS -m Bezier edit /tmp/option_d.py
```

Note that there was a small bug in the assumed sRGB viewing conditions
while designing viridis. It does not affect the outcome by much. Also
see `python -m viscm --help`.

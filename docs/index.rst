sdr
===

The :obj:`sdr` library is a Python 3 package for software-defined radio (SDR).

The goal of :obj:`sdr` is to provide tools to design, build, and analyze digital communications systems
in Python. The library relies on and is designed to be interoperable with `NumPy`_, `SciPy`_, and `Matplotlib`_.
Performance is also very important. So, where possible, `Numba`_ is used to accelerate computationally intensive
functions.

Additionally, the library aims to replicate relevant functionality from Matlab's `Communications`_ and
`DSP`_ Toolboxes.

We are progressively adding functionality to the library. If there is something you'd like to see included
in :obj:`sdr`, please open an issue on `GitHub`_.

.. _NumPy: https://numpy.org/
.. _SciPy: https://www.scipy.org/
.. _Matplotlib: https://matplotlib.org/
.. _Numba: https://numba.pydata.org/
.. _Communications: https://www.mathworks.com/products/communications.html
.. _DSP: https://www.mathworks.com/products/dsp-system.html
.. _GitHub: https://github.com/mhostetter/sdr/issues

Features
--------

- Filters: :obj:`sdr.IIR`
- Pulse shapes: :obj:`sdr.raised_cosine`, :obj:`sdr.root_raised_cosine`
- Arbitrary resamplers: :obj:`sdr.FarrowResampler`
- Phase-locked loops: :obj:`sdr.NCO`, :obj:`sdr.DDS`, :obj:`sdr.LoopFilter`, :obj:`sdr.ClosedLoopPLL`
- Data manipulation: :obj:`sdr.pack`, :obj:`sdr.unpack`, :obj:`sdr.hexdump`
- Plotting utilities: :obj:`sdr.plot.time_domain`, :obj:`sdr.plot.filter`, :obj:`sdr.plot.frequency_response`,
  :obj:`sdr.plot.group_delay`, :obj:`sdr.plot.impulse_response`, :obj:`sdr.plot.step_response`,
  :obj:`sdr.plot.zeros_poles`

.. toctree::
   :caption: Examples
   :hidden:

   examples/pulse-shapes.ipynb
   examples/iir-filter.ipynb
   examples/farrow-resampler.ipynb
   examples/phase-locked-loop.ipynb

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 2

   api.rst

.. toctree::
   :caption: Release Notes
   :hidden:

   release-notes/versioning.rst
   release-notes/v0.0.md

.. toctree::
   :caption: Index
   :hidden:

   genindex

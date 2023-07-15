sdr
===

The :obj:`sdr` library is a Python 3 package for software-defined radio (SDR) applications.


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

   examples/iir-filter.ipynb
   examples/raised-cosine-pulse.ipynb
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

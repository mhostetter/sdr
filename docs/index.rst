sdr
===

The :obj:`sdr` library is a Python 3 package for software-defined radio (SDR) applications.


Features
--------

- Filters:
   - Infinite impulse response filter (:obj:`sdr.IIR`)
- Arbitrary resamplers:
   - Farrow resampler (:obj:`sdr.FarrowResampler`)
- Signal generators:
   - Numerically-controlled oscillator (:obj:`sdr.NCO`)
   - Direct digital synthesizer (:obj:`sdr.DDS`)
- Phase-locked loops:
   - Loop filter (:obj:`sdr.LoopFilter`)
   - Closed-loop PLL analysis (:obj:`sdr.ClosedLoopPLL`)

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

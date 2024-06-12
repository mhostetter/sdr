sdr
===

.. raw:: html

   <div align=center>
      <a href="https://pypi.org/project/sdr"><img src="https://img.shields.io/pypi/v/sdr" class="no-invert"></a>
      <a href="https://pypi.org/project/sdr"><img src="https://img.shields.io/pypi/pyversions/sdr" class="no-invert"></a>
      <a href="https://pypi.org/project/sdr"><img src="https://img.shields.io/pypi/wheel/sdr" class="no-invert"></a>
      <a href="https://pypistats.org/packages/sdr"><img src="https://img.shields.io/pypi/dm/sdr" class="no-invert"></a>
      <a href="https://pypi.org/project/sdr"><img src="https://img.shields.io/pypi/l/sdr" class="no-invert"></a>
      <a href="https://twitter.com/sdr_py"><img src="https://img.shields.io/static/v1?label=follow&message=@sdr_py&color=blue&logo=twitter" class="no-invert"></a>
      <a href="https://github.com/mhostetter/sdr/actions/workflows/docs.yaml"><img src="https://github.com/mhostetter/sdr/actions/workflows/docs.yaml/badge.svg" class="no-invert"></a>
      <a href="https://github.com/mhostetter/sdr/actions/workflows/lint.yaml"><img src="https://github.com/mhostetter/sdr/actions/workflows/lint.yaml/badge.svg" class="no-invert"></a>
      <a href="https://github.com/mhostetter/sdr/actions/workflows/build.yaml"><img src="https://github.com/mhostetter/sdr/actions/workflows/build.yaml/badge.svg" class="no-invert"></a>
      <a href="https://github.com/mhostetter/sdr/actions/workflows/test.yaml"><img src="https://github.com/mhostetter/sdr/actions/workflows/test.yaml/badge.svg" class="no-invert"></a>
      <a href="https://codecov.io/gh/mhostetter/sdr"><img src="https://codecov.io/gh/mhostetter/sdr/branch/main/graph/badge.svg?token=3FJML79ZUK" class="no-invert"></a>
   </div>

The :obj:`sdr` library is a Python 3 package for software-defined radio (SDR).

The goal of :obj:`sdr` is to provide tools to design, analyze, build, and test digital communication systems
in Python. The library relies on and is designed to be interoperable with `NumPy`_, `SciPy`_, and `Matplotlib`_.
Performance is also very important. So, where possible, `Numba`_ is used to accelerate computationally intensive
functions.

Additionally, the library aims to replicate relevant functionality from MATLAB's `Communications`_ and
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

Installation
------------

The latest version of :obj:`sdr` can be installed from `PyPI`_ using `pip`_.

.. code-block:: bash

   python3 -m pip install sdr

.. _PyPI: https://pypi.org/project/sdr/
.. _pip: https://pip.pypa.io/en/stable/

Features
--------

View all available classes and functions in the `API Reference <https://mhostetter.github.io/sdr/latest/api/dsp/>`_.

- **Digital signal processing**: Finite impulse response (FIR) filters, FIR filter design,
  infinite impulse response (IIR) filters, polyphase interpolators, polyphase decimators, polyphase resamplers,
  polyphase channelizers, Farrow arbitrary resamplers, fractional delay FIR filters, FIR moving averagers,
  FIR differentiators, IIR integrators, IIR leaky integrators, complex mixing, real/complex conversion.
- **Sequences**: Binary, Gray, Barker, Hadamard, Walsh, Kasami, Zadoff-Chu, $m$-sequences, Fibonacci LFSRs, Galois LFSRs,
  LFSR synthesis.
- **Coding**: Block interleavers, additive scramblers.
- **Modulation**: Phase-shift keying (PSK), $\pi/M$ PSK, offset QPSK, continuous-phase modulation (CPM),
  minimum-shift keying (MSK), rectangular pulse shapes, half-sine pulse shapes, Gaussian pulse shapes,
  raised cosine pulse shapes, root raised cosine pulse shapes, differential encoding.
- **Estimation**: TOA, TDOA, FOA, FDOA Cramér-Rao lower bounds.
- **Detection**: Detector probability density functions (PDFs). Theoretical probability of detection, probability of
  false alarm, and thresholds. Detection performance approximations. Coherent gain, coherent gain loss (CGL),
  and non-coherent gain. Maximum-allowable integration time and frequency offset.
- **Synchronization**: Numerically controlled oscillators (NCO), loop filters, closed-loop phase-locked loop (PLL)
  analysis, phase error detectors (PEDs), automatic gain control (AGC).
- **Measurement**: Energy, power, voltage, Euclidean distance, Hamming distance, bit/symbol error rate,
  error vector magnitude (EVM), RMS integration time, and RMS bandwidth.
- **Conversions**: Between linear units and decibels. Between $E_b/N_0$, $E_s/N_0$, and $S/N$.
- **Simulation**: Binary symmetric channels (BSC), binary erasure channels (BEC), discrete memoryless channels (DMC).
  Apply additive white Gaussian noise (AWGN), frequency offset, sample rate offset, IQ imbalance.
- **Link budgets**: BSC, BEC, AWGN, and BI-AWGN channel capacity, Shannon's limit on $E_b/N_0$ and $S/N$,
  free-space path loss, antenna gain.
- **Miscellaneous**: Packing and unpacking binary data, hexdump of binary data.
- **Plotting**: Time-domain, raster, periodogram, spectrogram, constellation, symbol map, eye diagram, phase tree,
  bit error rate (BER), symbol error rate (SER), probability of detection, receiver operating characteristic (ROC),
  detection PDFs, impulse response, step response, zeros/poles, magnitude response, phase response,
  phase delay, and group delay.

.. toctree::
   :caption: Examples
   :hidden:

   examples/dsp.rst
   examples/modulation.rst
   examples/detection.rst
   examples/synchronization.rst

.. toctree::
   :caption: Development
   :hidden:

   development/installation.rst
   development/formatting.rst
   development/unit-tests.rst
   development/documentation.rst

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 4

   api/dsp.rst
   api/sequences.rst
   api/coding.rst
   api/modulation.rst
   api/estimation.rst
   api/detection.rst
   api/synchronization.rst
   api/measurement.rst
   api/conversions.rst
   api/simulation.rst
   api/link-budgets.rst
   api/misc.rst
   api/plotting.rst

.. toctree::
   :caption: Release Notes
   :hidden:

   release-notes/versioning.rst
   release-notes/v0.0.md

.. toctree::
   :caption: Index
   :hidden:

   genindex

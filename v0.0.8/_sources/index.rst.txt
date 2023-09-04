sdr
===

The :obj:`sdr` library is a Python 3 package for software-defined radio (SDR).

The goal of :obj:`sdr` is to provide tools to design, analyze, build, and test digital communication systems
in Python. The library relies on and is designed to be interoperable with `NumPy`_, `SciPy`_, and `Matplotlib`_.
Performance is also very important. So, where possible, `Numba`_ is used to accelerate computationally-intensive
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

- **Digital signal processing**: Finite impulse response (FIR) filters, infinite impulse response (IIR) filters,
  polyphase interpolator, polyphase decimator, polyphase resampler, Farrow arbitrary resampler,
  complex mixing, real/complex conversion.
- **Sequences**: Barker, Zadoff-Chu.
- **Modulation**: Phase-shift keying (PSK), $\pi/M$ PSK, offset QPSK, rectangular pulse shape, half-sine pulse shape,
  raised cosine pulse shape, root raised cosine pulse shape, Gaussian pulse shape, binary and Gray symbol mapping,
  differential encoding.
- **Synchronization**: Numerically-controlled oscillators (NCO), direct digital synthesizers (DDS), loop filters,
  closed-loop PLL analysis.
- **Measurement**: Energy, power, voltage, bit/symbol error rate, error vector magnitude (EVM).
- **Conversions**: Between linear units and decibels. Between $E_b/N_0$, $E_s/N_0$, and $S/N$.
- **Simulation**: Binary symmetric channel (BSC), binary erasure channel (BEC), discrete memoryless channel (DMC),
  additive white Gaussian noise (AWGN), frequency offset, sample rate offset, IQ imbalance.
- **Link budgets**: Channel capacities, free-space path loss, antenna gains.
- **Data manipulation**: Packing and unpacking binary data, hexdumping binary data.
- **Plotting**: Time-domain, raster, periodogram, spectrogram, constellation, symbol map, eye diagram,
  bit error rate (BER), symbol error rate (SER), impulse response, step response, magnitude response, phase response,
  phase delay, group delay, and zeros/poles.

.. toctree::
   :caption: Examples
   :hidden:

   examples/pulse-shapes.ipynb
   examples/peak-to-average-power.ipynb
   examples/fir-filters.ipynb
   examples/iir-filters.ipynb
   examples/farrow-resampler.ipynb
   examples/psk.ipynb
   examples/phase-locked-loop.ipynb

.. toctree::
   :caption: Development
   :hidden:

   development/installation.rst
   development/linter.rst
   development/unit-tests.rst
   development/documentation.rst

.. toctree::
   :caption: API Reference
   :hidden:
   :maxdepth: 4

   api/dsp.rst
   api/sequences.rst
   api/modulation.rst
   api/detection.rst
   api/synchronization.rst
   api/measurement.rst
   api/conversions.rst
   api/simulation.rst
   api/link-budgets.rst
   api/probability.rst
   api/data-manipulation.rst
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

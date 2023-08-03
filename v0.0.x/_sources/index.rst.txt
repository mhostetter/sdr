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

- Filtering: :class:`sdr.FIR`, :class:`sdr.FIRInterpolator`, :class:`sdr.IIR`
- Resampling: :class:`sdr.FarrowResampler`
- Sequences: :func:`sdr.barker()`, :func:`sdr.zadoff_chu()`
- Modulation:
   - Classes: :class:`sdr.PSK`
   - Pulse shapes: :func:`sdr.raised_cosine()`, :func:`sdr.root_raised_cosine()`, :func:`sdr.gaussian()`
   - Symbol mapping: :func:`sdr.binary_code()`, :func:`sdr.gray_code()`
   - Symbol encoding: :func:`sdr.diff_encode()`, :func:`sdr.diff_decode()`
- Synchronization: :class:`sdr.NCO`, :class:`sdr.DDS`, :class:`sdr.LoopFilter`, :class:`sdr.ClosedLoopPLL`
- Measurement:
   - Energy: :func:`sdr.energy()`
   - Power: :func:`sdr.peak_power()`, :func:`sdr.average_power()`, :func:`sdr.papr()`
   - Voltage: :func:`sdr.peak_voltage()`, :func:`sdr.rms_voltage()`, :func:`sdr.crest_factor()`
   - Modulation: :class:`sdr.ErrorRate`, :func:`sdr.evm()`
- Conversions:
   - From $E_b/N_0$: :func:`sdr.ebn0_to_esn0()`, :func:`sdr.ebn0_to_snr()`
   - From $E_s/N_0$: :func:`sdr.esn0_to_ebn0()`, :func:`sdr.esn0_to_snr()`
   - From $S/N$: :func:`sdr.snr_to_ebn0()`, :func:`sdr.snr_to_esn0()`
- Simulation:
   - Channel models: :func:`sdr.bec()`, :func:`sdr.bsc()`, :func:`sdr.dmc()`
   - Signal impairments: :func:`sdr.awgn()`, :func:`sdr.frequency_offset()`, :func:`sdr.sample_rate_offset()`,  :func:`sdr.iq_imbalance()`
- Link budgets:
   - Channel capacity: :func:`sdr.awgn_capacity()`, :func:`sdr.bec_capacity()`, :func:`sdr.bsc_capacity()`
   - Path losses: :func:`sdr.fspl()`
   - Antennas: :func:`sdr.parabolic_antenna()`
- Probability: :func:`sdr.Q()`, :func:`sdr.Qinv()`
- Data manipulation: :func:`sdr.pack()`, :func:`sdr.unpack()`, :func:`sdr.hexdump()`
- Plotting:
   - Time-domain: :func:`sdr.plot.time_domain()`
   - Frequency domain: :func:`sdr.plot.periodogram()`, :func:`sdr.plot.spectrogram()`
   - Modulation: :func:`sdr.plot.ber()`, :func:`sdr.plot.ser()`, :func:`sdr.plot.constellation()`, :func:`sdr.plot.symbol_map()`
   - Filters: :func:`sdr.plot.impulse_response()`, :func:`sdr.plot.step_response()`, :func:`sdr.plot.frequency_response()`,
     :func:`sdr.plot.phase_response()`, :func:`sdr.plot.phase_delay()`, :func:`sdr.plot.group_delay()`,
     :func:`sdr.plot.zeros_poles()`, :func:`sdr.plot.filter()`

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
   :caption: API Reference
   :hidden:
   :maxdepth: 4

   api/filtering.rst
   api/resampling.rst
   api/sequences.rst
   api/modulation.rst
   api/synchronization.rst
   api/impairments.rst
   api/measurement.rst
   api/conversions.rst
   api/simulation.rst
   api/link-budgets.rst
   api/probability.rst
   api/data-manipulation.rst
   api/plotting.rst

.. toctree::
   :caption: Development
   :hidden:

   development/installation.rst
   development/linter.rst
   development/unit-tests.rst
   development/documentation.rst

.. toctree::
   :caption: Release Notes
   :hidden:

   release-notes/versioning.rst
   release-notes/v0.0.md

.. toctree::
   :caption: Index
   :hidden:

   genindex

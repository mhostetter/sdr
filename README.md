# sdr

<div align=center>
  <a href="https://pypi.org/project/sdr"><img src="https://img.shields.io/pypi/v/sdr"></a>
  <a href="https://pypi.org/project/sdr"><img src="https://img.shields.io/pypi/pyversions/sdr"></a>
  <a href="https://pypi.org/project/sdr"><img src="https://img.shields.io/pypi/wheel/sdr"></a>
  <a href="https://pypistats.org/packages/sdr"><img src="https://img.shields.io/pypi/dm/sdr"></a>
  <a href="https://pypi.org/project/sdr"><img src="https://img.shields.io/pypi/l/sdr"></a>
  <a href="https://twitter.com/sdr_py"><img src="https://img.shields.io/static/v1?label=follow&message=@sdr_py&color=blue&logo=twitter"></a>
</div>

<div align=center>
  <a href="https://github.com/mhostetter/sdr/actions/workflows/docs.yaml"><img src="https://github.com/mhostetter/sdr/actions/workflows/docs.yaml/badge.svg"></a>
  <a href="https://github.com/mhostetter/sdr/actions/workflows/lint.yaml"><img src="https://github.com/mhostetter/sdr/actions/workflows/lint.yaml/badge.svg"></a>
  <a href="https://github.com/mhostetter/sdr/actions/workflows/build.yaml"><img src="https://github.com/mhostetter/sdr/actions/workflows/build.yaml/badge.svg"></a>
  <a href="https://github.com/mhostetter/sdr/actions/workflows/test.yaml"><img src="https://github.com/mhostetter/sdr/actions/workflows/test.yaml/badge.svg"></a>
  <a href="https://codecov.io/gh/mhostetter/sdr"><img src="https://codecov.io/gh/mhostetter/sdr/branch/main/graph/badge.svg?token=3FJML79ZUK"></a>
</div>

The `sdr` library is a Python 3 package for software-defined radio (SDR).

The goal of `sdr` is to provide tools to design, analyze, build, and test digital communication systems
in Python. The library relies on and is designed to be interoperable with NumPy, SciPy, and Matplotlib.
Performance is also very important. So, where possible, Numba is used to accelerate computationally-intensive
functions.

Additionally, the library aims to replicate relevant functionality from Matlab's Communications and
DSP Toolboxes.

We are progressively adding functionality to the library. If there is something you'd like to see included
in `sdr`, please open an issue on GitHub.

> Enjoying the library? Give us a :star: on [GitHub](https://github.com/mhostetter/sdr)!

## Documentation

The documentation for `sdr` is located at <https://mhostetter.github.io/sdr/latest/>.

## Installation

The latest version of `sdr` can be installed from [PyPI](https://pypi.org/project/sdr/) using `pip`.

```console
python3 -m pip install sdr
```

## Features

- Digital signal processing:
   - Filtering: `sdr.FIR`, `sdr.Interpolator`, `sdr.IIR`
   - Resampling: `sdr.FarrowResampler`
   - Signal manipulation: `sdr.mix()`, `sdr.to_complex_bb()`, `sdr.to_real_pb()`
- Sequences: `sdr.barker()`, `sdr.zadoff_chu()`
- Modulation:
   - Classes: `sdr.PSK`
   - Pulse shapes: `sdr.raised_cosine()`, `sdr.root_raised_cosine()`, `sdr.gaussian()`
   - Symbol mapping: `sdr.binary_code()`, `sdr.gray_code()`
   - Symbol encoding: `sdr.diff_encode()`, `sdr.diff_decode()`
- Synchronization: `sdr.NCO`, `sdr.DDS`, `sdr.LoopFilter`, `sdr.ClosedLoopPLL`
- Measurement:
   - Energy: `sdr.energy()`
   - Power: `sdr.peak_power()`, `sdr.average_power()`, `sdr.papr()`
   - Voltage: `sdr.peak_voltage()`, `sdr.rms_voltage()`, `sdr.crest_factor()`
   - Modulation: `sdr.ErrorRate`, `sdr.evm()`
- Conversions:
   - From $E_b/N_0$: `sdr.ebn0_to_esn0()`, `sdr.ebn0_to_snr()`
   - From $E_s/N_0$: `sdr.esn0_to_ebn0()`, `sdr.esn0_to_snr()`
   - From $S/N$: `sdr.snr_to_ebn0()`, `sdr.snr_to_esn0()`
- Simulation:
   - Channel models: `sdr.bec()`, `sdr.bsc()`, `sdr.dmc()`
   - Signal impairments: `sdr.awgn()`, `sdr.frequency_offset()`, `sdr.sample_rate_offset()`,
     `sdr.iq_imbalance()`
- Link budgets:
   - Channel capacity: `sdr.awgn_capacity()`, `sdr.bec_capacity()`, `sdr.bsc_capacity()`
   - Path losses: `sdr.fspl()`
   - Antennas: `sdr.parabolic_antenna()`
- Probability: `sdr.Q()`, `sdr.Qinv()`
- Data manipulation: `sdr.pack()`, `sdr.unpack()`, `sdr.hexdump()`
- Plotting:
   - Time-domain: `sdr.plot.time_domain()`
   - Spectral estimation: `sdr.plot.periodogram()`, `sdr.plot.spectrogram()`
   - Modulation: `sdr.plot.ber()`, `sdr.plot.ser()`, `sdr.plot.constellation()`,
     `sdr.plot.symbol_map()`
   - Filters: `sdr.plot.impulse_response()`, `sdr.plot.step_response()`,
     `sdr.plot.frequency_response()`, `sdr.plot.phase_response()`, `sdr.plot.phase_delay()`,
     `sdr.plot.group_delay()`, `sdr.plot.zeros_poles()`, `sdr.plot.filter()`

## Examples

There are detailed examples published at <https://mhostetter.github.io/sdr/latest/examples/pulse-shapes/>.
The Jupyter notebooks behind the examples are available for experimentation in `docs/examples/`.

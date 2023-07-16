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
  <a href="https://codecov.io/gh/mhostetter/sdr"><img src="https://codecov.io/gh/mhostetter/sdr/branch/master/graph/badge.svg?token=3FJML79ZUK"></a>
</div>

A Python package for software-defined radio.

The `sdr` library is a Python 3 package for software-defined radio (SDR).

The goal of `sdr` is to provide tools to design, build, and analyze digital communications systems
in Python. The library relies on and is designed to be interoperable with NumPy, SciPy, and Matplotlib.
Performance is also very important. So, where possible, Numba is used to accelerate computationally intensive
functions.

Additionally, the library aims to replicate relevant functionality from Matlab's Communications and
DSP Toolboxes.

We are progressively adding functionality to the library. If there is something you'd like to see included
in `sdr`, please open an issue on GitHub.

> Enjoying the library? Give us a :star: on [GitHub](https://github.com/mhostetter/sdr)!

## Features

- Filters: `sdr.FIR`, `sdr.IIR`
- Pulse shapes: `sdr.raised_cosine()`, `sdr.root_raised_cosine()`, `sdr.gaussian()`
- Arbitrary resamplers: `sdr.FarrowResampler`
- Phase-locked loops: `sdr.NCO`, `sdr.DDS`, `sdr.LoopFilter`, `sdr.ClosedLoopPLL`
- Impairments: `sdr.awgn()`
- Measurement: `sdr.peak_power()`, `sdr.average_power()`, `sdr.peak_voltage()`,
  `sdr.rms_voltage()`, `sdr.papr()`, `sdr.crest_factor()`
- Data manipulation: `sdr.pack()`, `sdr.unpack()`, `sdr.hexdump()`
- Plotting utilities: `sdr.plot.time_domain()`, `sdr.plot.periodogram()`, `sdr.plot.spectrogram()`
  `sdr.plot.filter()`, `sdr.plot.frequency_response`, `sdr.plot.group_delay()`,
  `sdr.plot.impulse_response()`, `sdr.plot.step_response()`, `sdr.plot.zeros_poles()`

## Documentation

The documentation for `sdr` is located at https://mhostetter.github.io/sdr/latest/.

## Install the package

The latest version of `sdr` can be installed from [PyPI](https://pypi.org/project/sdr/) using `pip`.

```console
$ python3 -m pip install sdr
```

## Examples

There are detailed examples published at https://mhostetter.github.io/sdr/latest/examples/pulse-shapes/.
The Jupyter notebooks behind the examples are available for experimentation in `docs/examples/`.

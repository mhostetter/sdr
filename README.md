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
Performance is also very important. So, where possible, Numba is used to accelerate computationally intensive
functions.

Additionally, the library aims to replicate relevant functionality from MATLAB's Communications and
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

View all available classes and functions in the [API Reference](https://mhostetter.github.io/sdr/latest/api/dsp/).

- **Digital signal processing**: Finite impulse response (FIR) filters, FIR filter design,
  infinite impulse response (IIR) filters, polyphase interpolators, polyphase decimators, polyphase resamplers,
  polyphase channelizers, Farrow arbitrary resamplers, fractional delay FIR filters, FIR moving averagers,
  FIR differentiators, IIR integrators, IIR leaky integrators, complex mixing, real/complex conversion.
- **Sequences**: Binary, Gray, Barker, Hadamard, Walsh, Kasami, Zadoff-Chu, m-sequences, Fibonacci LFSRs, Galois LFSRs,
  LFSR synthesis.
- **Coding**: Block interleavers, additive scramblers.
- **Modulation**: Phase-shift keying (PSK), $\pi/M$ PSK, offset QPSK, continuous-phase modulation (CPM),
  minimum-shift keying (MSK), rectangular pulse shapes, half-sine pulse shapes, Gaussian pulse shapes,
  raised cosine pulse shapes, root raised cosine pulse shapes, differential encoding.
- **Detection**: Theoretical probability of detection, probability of false alarm, and thresholds.
  Detection approximations. Coherent gain, coherent gain loss (CGL), non-coherent gain, maximum-allowable
  integration time and frequency offset.
- **Synchronization**: Numerically controlled oscillators (NCO), loop filters, closed-loop phase-locked loop (PLL)
  analysis, phase error detectors (PEDs), automatic gain control (AGC).
- **Measurement**: Energy, power, voltage, Euclidean distance, Hamming distance, bit/symbol error rate,
  error vector magnitude (EVM).
- **Conversions**: Between linear units and decibels. Between $E_b/N_0$, $E_s/N_0$, and $S/N$.
- **Simulation**: Binary symmetric channels (BSC), binary erasure channels (BEC), discrete memoryless channels (DMC).
  Apply additive white Gaussian noise (AWGN), frequency offset, sample rate offset, IQ imbalance.
- **Link budgets**: Channel capacity, free-space path loss, antenna gain.
- **Miscellaneous**: Packing and unpacking binary data, hexdump of binary data.
- **Plotting**: Time-domain, raster, periodogram, spectrogram, constellation, symbol map, eye diagram, phase tree,
  bit error rate (BER), symbol error rate (SER), probability of detection, receiver operating characteristic (ROC),
  detection PDFs, impulse response, step response, zeros/poles, magnitude response, phase response,
  phase delay, and group delay.

## Examples

There are detailed examples published at <https://mhostetter.github.io/sdr/latest/examples/pulse-shapes/>.
The Jupyter notebooks behind the examples are available for experimentation in `docs/examples/`.

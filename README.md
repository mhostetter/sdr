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

> Enjoying the library? Give us a :star: on [GitHub](https://github.com/mhostetter/sdr)!

## Features

- Filters:
   - Infinite impulse response filter (`sdr.IIR`)
- Arbitrary resamplers:
   - Farrow resampler (`sdr.FarrowResampler`)
- Signal generators:
   - Numerically-controlled oscillator (`sdr.NCO`)
   - Direct digital synthesizer (`sdr.DDS`)
- Phase-locked loops:
   - Loop filter (`sdr.LoopFilter`)
   - Closed-loop PLL analysis (`sdr.ClosedLoopPLL`)

## Documentation

The documentation for `sdr` is located at https://mhostetter.github.io/sdr/latest/.

## Install the package

The latest version of `sdr` can be installed from [PyPI](https://pypi.org/project/sdr/) using `pip`.

```console
$ python3 -m pip install sdr
```

## Examples

There are detailed examples published at https://mhostetter.github.io/sdr/latest/examples/iir-filter/.
The Jupyter notebooks behind the examples are available for experimentation in `docs/examples/`.

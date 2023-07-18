import numpy as np
import pytest

import sdr


def test_250_MHz():
    """
    Matlab:
        >> d = logspace(1, 6, 20); d'
        >> L = fspl(d, physconst('LightSpeed')/250e6); L'
    """
    d = np.logspace(1, 6, 20)
    fspl = sdr.fspl(d, 250e6)
    fspl_truth = np.array(
        [
            40.4065833953241,
            45.6697412900610,
            50.9328991847978,
            56.1960570795347,
            61.4592149742715,
            66.7223728690083,
            71.9855307637452,
            77.2486886584820,
            82.5118465532189,
            87.7750044479557,
            93.0381623426926,
            98.3013202374294,
            103.5644781321662,
            108.8276360269031,
            114.0907939216399,
            119.3539518163768,
            124.6171097111136,
            129.8802676058505,
            135.1434255005873,
            140.4065833953241,
        ]
    )
    assert isinstance(fspl, np.ndarray)
    assert np.allclose(fspl, fspl_truth)

    i = np.random.randint(0, d.size)
    fspl = sdr.fspl(d[i], 250e6)
    assert isinstance(fspl, float)
    assert fspl == pytest.approx(fspl_truth[i])


def test_7p5_GHz():
    """
    Matlab:
        >> d = logspace(1, 6, 20); d'
        >> L = fspl(d, physconst('LightSpeed')/7.5e9); L'
    """
    d = np.logspace(1, 6, 20)
    fspl = sdr.fspl(d, 7.5e9)
    fspl_truth = np.array(
        [
            69.9490084897174,
            75.2121663844542,
            80.4753242791911,
            85.7384821739279,
            91.0016400686647,
            96.2647979634016,
            101.5279558581384,
            106.7911137528753,
            112.0542716476121,
            117.3174295423489,
            122.5805874370858,
            127.8437453318227,
            133.1069032265595,
            138.3700611212963,
            143.6332190160332,
            148.8963769107700,
            154.1595348055068,
            159.4226927002437,
            164.6858505949805,
            169.9490084897174,
        ]
    )
    assert isinstance(fspl, np.ndarray)
    assert np.allclose(fspl, fspl_truth)

    i = np.random.randint(0, d.size)
    fspl = sdr.fspl(d[i], 7.5e9)
    assert isinstance(fspl, float)
    assert fspl == pytest.approx(fspl_truth[i])


def test_28p25_GHz():
    """
    Matlab:
        >> d = logspace(1, 6, 20); d'
        >> L = fspl(d, physconst('LightSpeed')/28.25e9); L'
    """
    d = np.logspace(1, 6, 20)
    fspl = sdr.fspl(d, 28.25e9)
    fspl_truth = np.array(
        [
            81.4681522649925,
            86.7313101597294,
            91.9944680544662,
            97.2576259492030,
            102.5207838439399,
            107.7839417386767,
            113.0470996334136,
            118.3102575281504,
            123.5734154228872,
            128.8365733176241,
            134.0997312123609,
            139.3628891070978,
            144.6260470018346,
            149.8892048965714,
            155.1523627913083,
            160.4155206860452,
            165.6786785807820,
            170.9418364755188,
            176.2049943702557,
            181.4681522649925,
        ]
    )
    assert isinstance(fspl, np.ndarray)
    assert np.allclose(fspl, fspl_truth)

    i = np.random.randint(0, d.size)
    fspl = sdr.fspl(d[i], 28.25e9)
    assert isinstance(fspl, float)
    assert fspl == pytest.approx(fspl_truth[i])

import numpy as np
import pytest

import sdr

from .helper import verify_impulse_response


def test_exceptions():
    with pytest.raises(TypeError):
        sdr.design_bandpass_fir("30", 0.4, 0.25)
    with pytest.raises(ValueError):
        sdr.design_bandpass_fir(29, 0.4, 0.25)

    with pytest.raises(TypeError):
        sdr.design_bandpass_fir(30, "0.4", 0.25)
    with pytest.raises(ValueError):
        sdr.design_bandpass_fir(30, -0.1, 0.25)

    with pytest.raises(TypeError):
        sdr.design_bandpass_fir(30, 0.4, "0.25")
    with pytest.raises(ValueError):
        sdr.design_bandpass_fir(30, 0.4, 0.61 * 2)
    with pytest.raises(ValueError):
        sdr.design_bandpass_fir(30, 0.4, 0.41 * 2)

    with pytest.raises(ValueError):
        sdr.design_bandpass_fir(30, 0.4, 0.25, window="invalid")
    with pytest.raises(ValueError):
        sdr.design_bandpass_fir(30, 0.4, 0.25, window=np.ones(10))


def test_custom():
    """
    MATLAB:
        >> h = designBandpassFIR(FilterOrder=30, CenterFrequency=0.4, Bandwidth=0.25, Window="custom", CustomWindow=ones(1, 31));
        >> transpose(h)
    """
    h = sdr.design_bandpass_fir(30, 0.4, 0.25, window=None)
    h_truth = np.array(
        [
            -0.017963811098946,
            -0.010989790375857,
            0.040483655096790,
            0.047470791892434,
            -0.018274903941445,
            -0.049789192200645,
            -0.009251871522192,
            -0.000000000000000,
            -0.031142203848832,
            0.025642844210333,
            0.130105429160949,
            0.054396687090110,
            -0.175429172086089,
            -0.201401513132611,
            0.083266843699732,
            0.276509440666166,
            0.083266843699732,
            -0.201401513132611,
            -0.175429172086089,
            0.054396687090110,
            0.130105429160949,
            0.025642844210333,
            -0.031142203848832,
            -0.000000000000000,
            -0.009251871522192,
            -0.049789192200645,
            -0.018274903941445,
            0.047470791892434,
            0.040483655096790,
            -0.010989790375857,
            -0.017963811098946,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_hamming():
    """
    MATLAB:
        >> h = designBandpassFIR(FilterOrder=30, CenterFrequency=0.4, Bandwidth=0.25, Window="hamming");
        >> transpose(h)
    """
    h = sdr.design_bandpass_fir(30, 0.4, 0.25, window="hamming")
    h_truth = np.array(
        [
            -0.001295920763505,
            -0.000892428133759,
            0.004372345234144,
            0.007185275937122,
            -0.003826547884793,
            -0.013918318028965,
            -0.003319260363488,
            -0.000000000000000,
            -0.016514978737540,
            0.015773739691351,
            0.090339176322020,
            0.041586836521494,
            -0.144296905913076,
            -0.174392736680514,
            0.074331760946352,
            0.249344587494965,
            0.074331760946352,
            -0.174392736680514,
            -0.144296905913076,
            0.041586836521494,
            0.090339176322020,
            0.015773739691351,
            -0.016514978737540,
            -0.000000000000000,
            -0.003319260363488,
            -0.013918318028965,
            -0.003826547884793,
            0.007185275937122,
            0.004372345234144,
            -0.000892428133759,
            -0.001295920763505,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_hann():
    """
    MATLAB:
        >> h = designBandpassFIR(FilterOrder=30, CenterFrequency=0.4, Bandwidth=0.25, Window="hann");
        >> transpose(h)
    """
    h = sdr.design_bandpass_fir(30, 0.4, 0.25, window="hann")
    h_truth = np.array(
        [
            0,
            -0.000107362890179,
            0.001564707596353,
            0.004053095706492,
            -0.002703194884773,
            -0.011129374154929,
            -0.002858002590150,
            -0.000000000000000,
            -0.015377718751261,
            0.015006421380158,
            0.087247368558493,
            0.040590876565245,
            -0.141876386618995,
            -0.172292816804829,
            0.073637062395384,
            0.247232532726566,
            0.073637062395384,
            -0.172292816804829,
            -0.141876386618995,
            0.040590876565245,
            0.087247368558493,
            0.015006421380158,
            -0.015377718751261,
            -0.000000000000000,
            -0.002858002590150,
            -0.011129374154929,
            -0.002703194884773,
            0.004053095706492,
            0.001564707596353,
            -0.000107362890179,
            0,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_blackman():
    """
    MATLAB:
        >> h = designBandpassFIR(FilterOrder=30, CenterFrequency=0.4, Bandwidth=0.25, Window="blackman");
        >> transpose(h)
    """
    h = sdr.design_bandpass_fir(30, 0.4, 0.25, window="blackman")
    h_truth = np.array(
        [
            0.000000000000000,
            -0.000041220652235,
            0.000634590332168,
            0.001785625081886,
            -0.001317506937803,
            -0.006054485538259,
            -0.001737510892287,
            -0.000000000000000,
            -0.011477784663031,
            0.012227956391224,
            0.076671645271867,
            0.037968922297635,
            -0.139356067333892,
            -0.175261307377343,
            0.076498342595344,
            0.258647793242859,
            0.076498342595344,
            -0.175261307377343,
            -0.139356067333892,
            0.037968922297635,
            0.076671645271867,
            0.012227956391224,
            -0.011477784663031,
            -0.000000000000000,
            -0.001737510892287,
            -0.006054485538259,
            -0.001317506937803,
            0.001785625081886,
            0.000634590332168,
            -0.000041220652235,
            0.000000000000000,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_blackman_harris():
    """
    MATLAB:
        >> h = designBandpassFIR(FilterOrder=30, CenterFrequency=0.4, Bandwidth=0.25, Window="blackman-harris");
        >> transpose(h)
    """
    h = sdr.design_bandpass_fir(30, 0.4, 0.25, window="blackman-harris")
    h_truth = np.array(
        [
            -0.000001060796132,
            -0.000008076310132,
            0.000143451692851,
            0.000513101416779,
            -0.000480260735282,
            -0.002726738086864,
            -0.000937988377148,
            -0.000000000000000,
            -0.008213858159922,
            0.009739012408174,
            0.066659327027275,
            0.035402070593523,
            -0.137060855858530,
            -0.178985506259001,
            0.079892784955629,
            0.272139862587763,
            0.079892784955629,
            -0.178985506259001,
            -0.137060855858530,
            0.035402070593523,
            0.066659327027275,
            0.009739012408174,
            -0.008213858159922,
            -0.000000000000000,
            -0.000937988377148,
            -0.002726738086864,
            -0.000480260735282,
            0.000513101416779,
            0.000143451692851,
            -0.000008076310132,
            -0.000001060796132,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_chebyshev():
    """
    MATLAB:
        >> h = designBandpassFIR(FilterOrder=30, CenterFrequency=0.4, Bandwidth=0.25, Window="chebyshev");
        >> transpose(h)
    """
    h = sdr.design_bandpass_fir(30, 0.4, 0.25, window="chebyshev")
    h_truth = np.array(
        [
            -0.000309113441909,
            -0.000349144761797,
            0.002354737618114,
            0.004540140197669,
            -0.002658020078420,
            -0.010356031164874,
            -0.002615046535495,
            -0.000000000000000,
            -0.014316841728137,
            0.014231877095468,
            0.084372501258072,
            0.039973462497828,
            -0.141881452704038,
            -0.174303969847392,
            0.075032265565100,
            0.252528196624519,
            0.075032265565100,
            -0.174303969847392,
            -0.141881452704038,
            0.039973462497828,
            0.084372501258072,
            0.014231877095468,
            -0.014316841728137,
            -0.000000000000000,
            -0.002615046535495,
            -0.010356031164874,
            -0.002658020078420,
            0.004540140197669,
            0.002354737618114,
            -0.000349144761797,
            -0.000309113441909,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_kaiser():
    """
    MATLAB:
        >> h = designBandpassFIR(FilterOrder=30, CenterFrequency=0.4, Bandwidth=0.25, Window="kaiser");
        >> transpose(h)
    """
    h = sdr.design_bandpass_fir(30, 0.4, 0.25, window="kaiser")
    h_truth = np.array(
        [
            -0.016765450319470,
            -0.010339454228757,
            0.038373027043513,
            0.045306490646877,
            -0.017552083072319,
            -0.048095293336892,
            -0.008983538311274,
            -0.000000000000000,
            -0.030503119574613,
            0.025205246008597,
            0.128266302246497,
            0.053758312599290,
            -0.173698261663486,
            -0.199683408020401,
            0.082623297525045,
            0.274446319276530,
            0.082623297525045,
            -0.199683408020401,
            -0.173698261663486,
            0.053758312599290,
            0.128266302246497,
            0.025205246008597,
            -0.030503119574613,
            -0.000000000000000,
            -0.008983538311274,
            -0.048095293336892,
            -0.017552083072319,
            0.045306490646877,
            0.038373027043513,
            -0.010339454228757,
            -0.016765450319470,
        ]
    )
    verify_impulse_response(h, h_truth)

import numpy as np

import sdr

from ..fir.helper import verify_impulse_response


def test_0p25_10():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.25, 10);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(10, 0.25)
    h_truth = np.array(
        [
            0.003897161584427,
            -0.018854511975166,
            0.054685571980477,
            -0.146916834843871,
            0.885180083891852,
            0.295060027963951,
            -0.104940596317051,
            0.044742740711300,
            -0.016340577045144,
            0.003486934049224,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p25_21():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.25, 21);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(21, 0.25)
    h_truth = np.array(
        [
            -0.000459853432853,
            0.001780301221107,
            -0.004411475554865,
            0.009032428265480,
            -0.016593109078330,
            0.028617764027093,
            -0.048160636016254,
            0.083433312698903,
            -0.168989696305082,
            0.895499667695699,
            0.298499889231900,
            -0.120706925932201,
            0.068263619480921,
            -0.041739217880753,
            0.025605367813715,
            -0.015150230028040,
            0.008363359505074,
            -0.004126864228744,
            0.001678569722758,
            -0.000436271205527,
            0,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p25_30():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.25, 30);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(30, 0.25)
    h_truth = np.array(
        [
            0.000162561838444,
            -0.000508395607218,
            0.001127988095460,
            -0.002135243496399,
            0.003669453191154,
            -0.005901583587943,
            0.009047797809028,
            -0.013398371875938,
            0.019380078286091,
            -0.027696831498325,
            0.039676604805645,
            -0.058266361293686,
            0.091685802021204,
            -0.174618165106212,
            0.897918286371997,
            0.299306095457332,
            -0.124727260790152,
            0.075015656199167,
            -0.050497513121195,
            0.035500120089261,
            -0.025288411368036,
            0.017944516931565,
            -0.012533960787168,
            0.008530780791369,
            -0.005598938275740,
            0.003498780949705,
            -0.002044382071020,
            0.001083753268187,
            -0.000489908494228,
            0.000157051267649,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p5_10():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.5, 10);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(10, 0.5)
    h_truth = np.array(
        [
            0.005176009486030,
            -0.024620725533890,
            0.069212653991800,
            -0.172171217964805,
            0.622403280020865,
            0.622403280020865,
            -0.172171217964805,
            0.069212653991800,
            -0.024620725533890,
            0.005176009486030,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p5_21():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.5, 21);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(21, 0.5)
    h_truth = np.array(
        [
            -0.000632089078103,
            0.002439322594027,
            -0.006020067295678,
            0.012260605189456,
            -0.022359655494891,
            0.038155121464370,
            -0.063131812228259,
            0.106004114561687,
            -0.198801522142286,
            0.632085982429676,
            0.632085982429676,
            -0.198801522142286,
            0.106004114561687,
            -0.063131812228259,
            0.038155121464370,
            -0.022359655494891,
            0.012260605189456,
            -0.006020067295678,
            0.002439322594027,
            -0.000632089078103,
            0,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p5_30():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.5, 30);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(30, 0.5)
    h_truth = np.array(
        [
            0.000225731894765,
            -0.000705036111597,
            0.001561917369853,
            -0.002951410055467,
            0.005061315161735,
            -0.008119217076014,
            0.012408102500694,
            -0.018300216746842,
            0.026329926725985,
            -0.037355466604570,
            0.052946637054160,
            -0.076447040434613,
            0.116592888167806,
            -0.205605930369026,
            0.634357798523132,
            0.634357798523132,
            -0.205605930369026,
            0.116592888167806,
            -0.076447040434613,
            0.052946637054160,
            -0.037355466604570,
            0.026329926725985,
            -0.018300216746842,
            0.012408102500694,
            -0.008119217076014,
            0.005061315161735,
            -0.002951410055467,
            0.001561917369853,
            -0.000705036111597,
            0.000225731894765,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p75_10():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.75, 10);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(10, 0.75)
    h_truth = np.array(
        [
            0.003486934049224,
            -0.016340577045144,
            0.044742740711300,
            -0.104940596317051,
            0.295060027963951,
            0.885180083891852,
            -0.146916834843871,
            0.054685571980477,
            -0.018854511975166,
            0.003897161584427,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p75_21():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.75, 21);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(21, 0.75)
    h_truth = np.array(
        [
            -0.000436271205527,
            0.001678569722758,
            -0.004126864228744,
            0.008363359505074,
            -0.015150230028040,
            0.025605367813715,
            -0.041739217880753,
            0.068263619480921,
            -0.120706925932201,
            0.298499889231900,
            0.895499667695699,
            -0.168989696305082,
            0.083433312698903,
            -0.048160636016254,
            0.028617764027093,
            -0.016593109078330,
            0.009032428265480,
            -0.004411475554865,
            0.001780301221107,
            -0.000459853432853,
            0,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p75_30():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.75, 30);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(30, 0.75)
    h_truth = np.array(
        [
            0.000157051267649,
            -0.000489908494228,
            0.001083753268187,
            -0.002044382071020,
            0.003498780949705,
            -0.005598938275740,
            0.008530780791369,
            -0.012533960787168,
            0.017944516931565,
            -0.025288411368036,
            0.035500120089261,
            -0.050497513121195,
            0.075015656199167,
            -0.124727260790152,
            0.299306095457332,
            0.897918286371997,
            -0.174618165106213,
            0.091685802021204,
            -0.058266361293686,
            0.039676604805645,
            -0.027696831498325,
            0.019380078286091,
            -0.013398371875938,
            0.009047797809028,
            -0.005901583587943,
            0.003669453191154,
            -0.002135243496399,
            0.001127988095460,
            -0.000508395607218,
            0.000162561838444,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p123456789_10():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.123456789, 10);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(10, 0.123456789)
    h_truth = np.array(
        [
            0.002163817437857,
            -0.010568366744468,
            0.031214513496624,
            -0.088058224499761,
            0.965609181690984,
            0.136001291783990,
            -0.052719068530707,
            0.023042473461296,
            -0.008515276384121,
            0.001829658288306,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p123456789_21():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.123456789, 21);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(21, 0.123456789)
    h_truth = np.array(
        [
            -0.000249932079617,
            0.000969229937854,
            -0.002406862863610,
            0.004942061409987,
            -0.009114739279314,
            0.015811877360630,
            -0.026863359814544,
            0.047391323177794,
            -0.100793870422852,
            0.972099494329011,
            0.136915420315067,
            -0.060343698643526,
            0.034984152699933,
            -0.021644681619954,
            0.013370043128595,
            -0.007946673948241,
            0.004400830266613,
            -0.002176739611106,
            0.000887000415771,
            -0.000230874758491,
            0,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p123456789_30():
    """
    MATLAB:
        >> h = designFracDelayFIR(0.123456789, 30);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(30, 0.123456789)
    h_truth = np.array(
        [
            0.000087825242290,
            -0.000274850364771,
            0.000610296714112,
            -0.001156344006638,
            0.001989405341252,
            -0.003203891288677,
            0.004920201129572,
            -0.007301736150528,
            0.010591702712345,
            -0.015196837170653,
            0.021897214852184,
            -0.032463299009055,
            0.052019727669177,
            -0.104032720580176,
            0.973618333805220,
            0.137129341365833,
            -0.062282747089877,
            0.038400828973691,
            -0.026156734534893,
            0.018515619637853,
            -0.013249343326118,
            0.009431749629266,
            -0.006603607770516,
            0.004502770990822,
            -0.002959594577214,
            0.001851659908601,
            -0.001083020737814,
            0.000574603425839,
            -0.000259934108277,
            0.000083379317149,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p876543211_10():
    """
    MATLAB:
        >> h = designFracDelayFIR(1 - 0.123456789, 10);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(10, 1 - 0.123456789)
    h_truth = np.array(
        [
            0.001829658288306,
            -0.008515276384121,
            0.023042473461296,
            -0.052719068530707,
            0.136001291783990,
            0.965609181690984,
            -0.088058224499761,
            0.031214513496624,
            -0.010568366744468,
            0.002163817437857,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p876543211_21():
    """
    MATLAB:
        >> h = designFracDelayFIR(1 - 0.123456789, 21);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(21, 1 - 0.123456789)
    h_truth = np.array(
        [
            -0.000230874758491,
            0.000887000415771,
            -0.002176739611106,
            0.004400830266613,
            -0.007946673948242,
            0.013370043128595,
            -0.021644681619954,
            0.034984152699933,
            -0.060343698643526,
            0.136915420315067,
            0.972099494329011,
            -0.100793870422852,
            0.047391323177794,
            -0.026863359814544,
            0.015811877360630,
            -0.009114739279315,
            0.004942061409988,
            -0.002406862863610,
            0.000969229937854,
            -0.000249932079617,
            0,
        ]
    )
    verify_impulse_response(h, h_truth)


def test_0p876543211_30():
    """
    MATLAB:
        >> h = designFracDelayFIR(1 - 0.123456789, 30);
        >> transpose(h)
    """
    h = sdr.fractional_delay_fir(30, 1 - 0.123456789)
    h_truth = np.array(
        [
            0.000083379317149,
            -0.000259934108277,
            0.000574603425839,
            -0.001083020737814,
            0.001851659908601,
            -0.002959594577214,
            0.004502770990822,
            -0.006603607770516,
            0.009431749629266,
            -0.013249343326119,
            0.018515619637853,
            -0.026156734534893,
            0.038400828973691,
            -0.062282747089877,
            0.137129341365833,
            0.973618333805220,
            -0.104032720580176,
            0.052019727669177,
            -0.032463299009055,
            0.021897214852185,
            -0.015196837170653,
            0.010591702712345,
            -0.007301736150528,
            0.004920201129572,
            -0.003203891288677,
            0.001989405341252,
            -0.001156344006638,
            0.000610296714112,
            -0.000274850364771,
            0.000087825242290,
        ]
    )
    verify_impulse_response(h, h_truth)
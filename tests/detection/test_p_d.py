"""
These test vectors were manually verified through simulation.
"""

import numpy as np

import sdr


def test_real_coherent():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="coherent", complex=False)
    p_d_truth = np.array(
        [
            [
                0.00276855,
                0.00311505,
                0.00354992,
                0.0041021,
                0.00481212,
                0.0057375,
                0.0069609,
                0.00860275,
                0.01084068,
                0.01393962,
                0.01829847,
                0.02452172,
                0.03352737,
                0.04670335,
                0.06611846,
                0.09476801,
                0.13676489,
                0.19724273,
                0.28151531,
                0.39286892,
                0.52871709,
            ]
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_real_coherent_matlab():
    """
    MATLAB:
        >> [pd, pfa] = rocsnr(0, 'NumPoints', 11, 'SignalType', 'real');
        >> [pd, pfa] = rocsnr(5, 'NumPoints', 11, 'SignalType', 'real');
        >> [pd, pfa] = rocsnr(10, 'NumPoints', 11, 'SignalType', 'real');
        >> [pd, pfa] = rocsnr(15, 'NumPoints', 11, 'SignalType', 'real');
        >> [pd, pfa] = rocsnr(20, 'NumPoints', 11, 'SignalType', 'real');
    """
    p_fa = np.logspace(-10, 0, 11)

    p_d = sdr.p_d(0, p_fa, detector="coherent", complex=False)
    p_d_truth = np.array(
        [
            0.000000041303229,
            0.000000289929863,
            0.000001994053036,
            0.000013384848316,
            0.000087217610140,
            0.000547531437809,
            0.003273817235028,
            0.018298468405657,
            0.092362248073694,
            0.389143691645361,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(5, p_fa, detector="coherent", complex=False)
    p_d_truth = np.array(
        [
            0.000002291086087,
            0.000012240737357,
            0.000063109434162,
            0.000311889941350,
            0.001464250920361,
            0.006448309557579,
            0.026145088439151,
            0.094768014125701,
            0.291822446755110,
            0.690309507929810,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(10, p_fa, detector="coherent", complex=False)
    p_d_truth = np.array(
        [
            0.000689374601387,
            0.002287490523630,
            0.007148295954364,
            0.020822018758840,
            0.055788288155516,
            0.135097602110355,
            0.288852942986193,
            0.528717092834614,
            0.798402797776116,
            0.969995405956180,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(15, p_fa, detector="coherent", complex=False)
    p_d_truth = np.array(
        [
            0.230279207680923,
            0.354055691217959,
            0.504552633569146,
            0.664244685617704,
            0.807846776640121,
            0.912851019315182,
            0.971570734988738,
            0.994348370608612,
            0.999511496219396,
            0.999992935976652,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(20, p_fa, detector="coherent", complex=False)
    p_d_truth = np.array(
        [
            0.999862969351070,
            0.999968620962137,
            0.999994280078496,
            0.999999209291716,
            0.999999922523898,
            0.999999995127523,
            0.999999999831781,
            0.999999999997573,
            0.999999999999992,
            1.000000000000000,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_coherent():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="coherent")
    p_d_truth = np.array(
        [
            0.00410852,
            0.00482044,
            0.00574841,
            0.00697543,
            0.0086224,
            0.01086766,
            0.01397727,
            0.0183518,
            0.02459838,
            0.03363894,
            0.04686726,
            0.06636041,
            0.09512449,
            0.13728417,
            0.19798103,
            0.28252208,
            0.39415521,
            0.53020989,
            0.67799934,
            0.81475691,
            0.91649936,
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_coherent_matlab():
    """
    MATLAB:
        >> [pd, pfa] = rocsnr(0, 'NumPoints', 11);
        >> [pd, pfa] = rocsnr(5, 'NumPoints', 11);
        >> [pd, pfa] = rocsnr(10, 'NumPoints', 11);
        >> [pd, pfa] = rocsnr(15, 'NumPoints', 11);
        >> [pd, pfa] = rocsnr(20, 'NumPoints', 11);
    """
    p_fa = np.logspace(-10, 0, 11)

    p_d = sdr.p_d(0, p_fa, detector="coherent", complex=True)
    p_d_truth = np.array(
        [
            0.000000376583542,
            0.000002285262360,
            0.000013476757436,
            0.000076815971583,
            0.000420083990888,
            0.002181311489748,
            0.010588806665155,
            0.046867260615355,
            0.180849009034276,
            0.552769650359353,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(5, p_fa, detector="coherent", complex=True)
    p_d_truth = np.array(
        [
            0.000059914909502,
            0.000247969606921,
            0.000977006350804,
            0.003632237274751,
            0.012592362729097,
            0.040057091897662,
            0.114265875939688,
            0.282522083834916,
            0.574765082904296,
            0.891270922953047,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(10, p_fa, detector="coherent", complex=True)
    p_d_truth = np.array(
        [
            0.029432186033950,
            0.063545902644344,
            0.127171214031724,
            0.233551224778505,
            0.389244621079371,
            0.582090800301111,
            0.774310937108565,
            0.916499356277382,
            0.984055054999439,
            0.999290073159946,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(15, p_fa, detector="coherent", complex=True)
    p_d_truth = np.array(
        [
            0.944236427637412,
            0.974702578808063,
            0.990376342629301,
            0.997050737794405,
            0.999311150657931,
            0.999886906664244,
            0.999988505651821,
            0.999999420364825,
            0.999999990797367,
            0.999999999987310,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(20, p_fa, detector="coherent", complex=True)
    p_d_truth = np.array(
        [
            0.999999999999996,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_real_linear():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="linear", complex=False)
    p_d_truth = np.array(
        [
            [
                0.00162332,
                0.00179743,
                0.00202417,
                0.00232169,
                0.00271559,
                0.00324256,
                0.00395603,
                0.00493509,
                0.00629867,
                0.00822823,
                0.01100431,
                0.01506513,
                0.02109971,
                0.0301926,
                0.04403871,
                0.06523568,
                0.09761458,
                0.14644439,
                0.21809582,
                0.31841208,
                0.44897593,
            ]
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_linear():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="linear")
    p_d_truth = np.array(
        [
            0.00177786,
            0.0020086,
            0.00231693,
            0.00273403,
            0.00330629,
            0.00410404,
            0.00523576,
            0.00687179,
            0.00928377,
            0.01291049,
            0.01846703,
            0.02712174,
            0.0407706,
            0.06242657,
            0.09667487,
            0.14995286,
            0.23001292,
            0.34340965,
            0.48995306,
            0.65552537,
            0.81029237,
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_real_square_law():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="square-law", complex=False)
    p_d_truth = np.array(
        [
            [
                0.00162332,
                0.00179743,
                0.00202417,
                0.00232169,
                0.00271559,
                0.00324256,
                0.00395603,
                0.00493509,
                0.00629867,
                0.00822823,
                0.01100431,
                0.01506513,
                0.02109971,
                0.0301926,
                0.04403871,
                0.06523568,
                0.09761458,
                0.14644439,
                0.21809582,
                0.31841208,
                0.44897593,
            ]
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_square_law():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="square-law")
    p_d_truth = np.array(
        [
            0.00177786,
            0.0020086,
            0.00231693,
            0.00273403,
            0.00330629,
            0.00410404,
            0.00523576,
            0.00687179,
            0.00928377,
            0.01291049,
            0.01846703,
            0.02712174,
            0.0407706,
            0.06242657,
            0.09667487,
            0.14995286,
            0.23001292,
            0.34340965,
            0.48995306,
            0.65552537,
            0.81029237,
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_square_law_matlab():
    """
    MATLAB:
        >> [pd, pfa] = rocsnr(0, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent');
        >> [pd, pfa] = rocsnr(5, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent');
        >> [pd, pfa] = rocsnr(10, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent');
        >> [pd, pfa] = rocsnr(15, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent');
        >> [pd, pfa] = rocsnr(20, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent');
    """
    p_fa = np.logspace(-10, 0, 11)

    p_d = sdr.p_d(0, p_fa, detector="square-law", complex=True)
    p_d_truth = np.array(
        [
            0.000000087577094,
            0.000000556273704,
            0.000003453282067,
            0.000020869580832,
            0.000122143702230,
            0.000687376042475,
            0.003681298651364,
            0.018467034586841,
            0.084477430252036,
            0.334373155416732,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(5, p_fa, detector="square-law", complex=True)
    p_d_truth = np.array(
        [
            0.000016330225745,
            0.000071727798406,
            0.000302129560773,
            0.001211808846395,
            0.004585348356211,
            0.016163484412142,
            0.052144218590045,
            0.149952862534075,
            0.368772236335902,
            0.722916566487456,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(10, p_fa, detector="square-law", complex=True)
    p_d_truth = np.array(
        [
            0.013093657826321,
            0.030563740425308,
            0.066625820239989,
            0.134380042690888,
            0.248049275738192,
            0.413860313031628,
            0.616135848507555,
            0.810292374261240,
            0.942251421470721,
            0.993561010011120,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(15, p_fa, detector="square-law", complex=True)
    p_d_truth = np.array(
        [
            0.891542526063240,
            0.943493264897015,
            0.974708754869056,
            0.990597154098270,
            0.997225393230807,
            0.999391019426204,
            0.999909968554916,
            0.999992412186489,
            0.999999736960102,
            0.999999998397952,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(20, p_fa, detector="square-law", complex=True)
    p_d_truth = np.array(
        [
            0.999999999999935,
            0.999999999999996,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_square_law_non_coherent_matlab():
    """
    MATLAB:
        >> [pd, pfa] = rocsnr(0, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent', 'NumPulses', 2);
        >> [pd, pfa] = rocsnr(0, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent', 'NumPulses', 3);
        >> [pd, pfa] = rocsnr(0, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent', 'NumPulses', 4);
        >> [pd, pfa] = rocsnr(0, 'NumPoints', 11, 'SignalType', 'NonfluctuatingNoncoherent', 'NumPulses', 5);
    """
    p_fa = np.logspace(-10, 0, 11)

    p_d = sdr.p_d(0, p_fa, detector="square-law", complex=True, n_nc=2)
    p_d_truth = np.array(
        [
            0.000000510660981,
            0.000002829338059,
            0.000015251861942,
            0.000079632665811,
            0.000400326053786,
            0.001922075517200,
            0.008711920577472,
            0.036619378144659,
            0.138510526442168,
            0.443895692335610,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(0, p_fa, detector="square-law", complex=True, n_nc=3)
    p_d_truth = np.array(
        [
            0.000001774064266,
            0.000008921067898,
            0.000043505697029,
            0.000204738238734,
            0.000923706322429,
            0.003959773235832,
            0.015923800523315,
            0.058904319583881,
            0.193907122007871,
            0.531903719032655,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(0, p_fa, detector="square-law", complex=True, n_nc=4)
    p_d_truth = np.array(
        [
            0.000004747839254,
            0.000022092630264,
            0.000099434362587,
            0.000430556712668,
            0.001780977374369,
            0.006969939314734,
            0.025453187781957,
            0.084925087405211,
            0.249889558515359,
            0.605271516968452,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

    p_d = sdr.p_d(0, p_fa, detector="square-law", complex=True, n_nc=5)
    p_d_truth = np.array(
        [
            0.000010801404553,
            0.000047061016205,
            0.000197875790460,
            0.000798339162580,
            0.003067424961146,
            0.011109414980930,
            0.037373891101969,
            0.114215123170587,
            0.305557842804910,
            0.667117395981415,
            1.000000000000000,
        ]
    )
    assert np.allclose(p_d, p_d_truth)

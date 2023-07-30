import numpy as np

import sdr


def test_bpsk_ser():
    """
    Matlab:
        >> ebn0 = 0:0.5:10;
        >> [ber, ser] = berawgn(ebn0, 'psk', 2, 'nondiff');
    """
    psk = sdr.PSK(2)

    ebn0 = np.arange(0, 10.5, 0.5)
    ser = psk.symbol_error_rate(ebn0)
    ser_truth = np.array(
        [
            0.078649603525143,
            0.067065198329613,
            0.056281951976541,
            0.046401275956071,
            0.037506128358926,
            0.029655287626037,
            0.022878407561085,
            0.017172541679246,
            0.012500818040738,
            0.008793810530561,
            0.005953867147779,
            0.003862231642810,
            0.002388290780933,
            0.001399804839480,
            0.000772674815378,
            0.000398796335159,
            0.000190907774076,
            0.000083999539179,
            0.000033627228420,
            0.000012108893277,
            0.000003872108216,
        ]
    )
    np.testing.assert_almost_equal(ser, ser_truth)


def test_qpsk_ser():
    """
    Matlab:
        >> ebn0 = 0:0.5:10;
        >> [ber, ser] = berawgn(ebn0, 'psk', 4, 'nondiff');
    """
    psk = sdr.PSK(4)

    ebn0 = np.arange(0, 10.5, 0.5)
    esn0 = sdr.ebn0_to_esn0(ebn0, psk.bps)
    ser = psk.symbol_error_rate(esn0)
    ser_truth = np.array(
        [
            0.151113446915623,
            0.129632655832235,
            0.109396245834793,
            0.090649473501791,
            0.073605547053376,
            0.058431139167891,
            0.045233393589640,
            0.034050187170767,
            0.024845365629788,
            0.017510289957474,
            0.011872285761544,
            0.007709546452358,
            0.004770877629011,
            0.002797650225372,
            0.001544752604387,
            0.000797433631801,
            0.000381779102374,
            0.000167992022435,
            0.000067253326049,
            0.000024217639929,
            0.000007744201438,
        ]
    )
    np.testing.assert_almost_equal(ser, ser_truth)


def test_8psk_ser():
    """
    Matlab:
        >> ebn0 = 0:0.5:10;
        >> [ber, ser] = berawgn(ebn0, 'psk', 8, 'nondiff');
    """
    psk = sdr.PSK(8)

    ebn0 = np.arange(0, 10.5, 0.5)
    esn0 = sdr.ebn0_to_esn0(ebn0, psk.bps)
    ser = psk.symbol_error_rate(esn0)
    ser_truth = np.array(
        [
            0.347800871199895,
            0.320262230592363,
            0.292616128934030,
            0.265075851862023,
            0.237871592637444,
            0.211246727844741,
            0.185452969086007,
            0.160744207836485,
            0.137368903756424,
            0.115560985581131,
            0.095529453105280,
            0.077447176298401,
            0.061439739725131,
            0.047575507975520,
            0.035858307286818,
            0.026224145471455,
            0.018543155232650,
            0.012627405490587,
            0.008244400588842,
            0.005135084035396,
            0.003034185962138,
        ]
    )

    # import matplotlib.pyplot as plt

    # plt.figure()
    # sdr.plot.ser(esn0, ser, label="Me")
    # sdr.plot.ser(esn0, ser_truth, label="Matlab")
    # plt.show()

    np.testing.assert_almost_equal(ser, ser_truth, decimal=3)


def test_16psk_ser():
    """
    Matlab:
        >> ebn0 = 0:0.5:10;
        >> [ber, ser] = berawgn(ebn0, 'psk', 16, 'nondiff');
    """
    psk = sdr.PSK(16)

    ebn0 = np.arange(0, 10.5, 0.5)
    esn0 = sdr.ebn0_to_esn0(ebn0, psk.bps)
    ser = psk.symbol_error_rate(esn0)
    ser_truth = np.array(
        [
            0.580976792181077,
            0.558826055301207,
            0.535799440104755,
            0.511927850919973,
            0.487252691121724,
            0.461827792445486,
            0.435721495259769,
            0.409018736505945,
            0.381822986042064,
            0.354257867520100,
            0.326468296781818,
            0.298620960546231,
            0.270903939013563,
            0.243525254285281,
            0.216710114353729,
            0.190696633868739,
            0.165729861047883,
            0.142054036040723,
            0.119903157906526,
            0.099490147901799,
            0.080995159210313,
        ]
    )

    # import matplotlib.pyplot as plt

    # plt.figure()
    # sdr.plot.ser(esn0, ser, label="Me")
    # sdr.plot.ser(esn0, ser_truth, label="Matlab")
    # plt.show()

    np.testing.assert_almost_equal(ser, ser_truth, decimal=3)

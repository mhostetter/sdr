import numpy as np

import sdr

# TODO: Commented-out tests are failing. Can't use approximation equations from Proakis.
#       Need to numerically integration equations to get exact values.


def test_bpsk_ber():
    """
    Matlab:
        >> ebn0 = 0:0.5:10;
        >> [ber, ser] = berawgn(ebn0, 'psk', 2, 'nondiff');
    """
    psk = sdr.PSK(2)

    ebn0 = np.arange(0, 10.5, 0.5)
    ber = psk.bit_error_rate(ebn0)
    ber_truth = np.array(
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
    assert np.allclose(ber, ber_truth)


def test_qpsk_ber():
    """
    Matlab:
        >> ebn0 = 0:0.5:10;
        >> [ber, ser] = berawgn(ebn0, 'psk', 4, 'nondiff');
    """
    psk = sdr.PSK(4)

    ebn0 = np.arange(0, 10.5, 0.5)
    ber = psk.bit_error_rate(ebn0)
    ber_truth = np.array(
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
    assert np.allclose(ber, ber_truth)


# def test_8psk_ber():
#     """
#     Matlab:
#         >> ebn0 = 0:0.5:10;
#         >> [ber, ser] = berawgn(ebn0, 'psk', 8, 'nondiff');
#     """
#     psk = sdr.PSK(8)

#     ebn0 = np.arange(0, 10.5, 0.5)
#     ber = psk.bit_error_rate(ebn0)
#     ber_truth = np.array(
#         [
#             0.122692761078506,
#             0.111540652877562,
#             0.100798515869953,
#             0.090483569561722,
#             0.080609413550440,
#             0.071190383412915,
#             0.062245644984332,
#             0.053802047467505,
#             0.045894918465536,
#             0.038566394957631,
#             0.031861441420881,
#             0.025822238602581,
#             0.020481966282913,
#             0.015859067150679,
#             0.011952902270821,
#             0.008741408289346,
#             0.006181056083768,
#             0.004209135737169,
#             0.002748133589174,
#             0.001711694683183,
#             0.001011395320989,
#         ]
#     )
#     assert np.allclose(ber, ber_truth)

# def test_16psk_ber():
#     """
#     Matlab:
#         >> ebn0 = 0:0.5:10;
#         >> [ber, ser] = berawgn(ebn0, 'psk', 16, 'nondiff');
#     """
#     psk = sdr.PSK(16)

#     ebn0 = np.arange(0, 10.5, 0.5)
#     ber = psk.bit_error_rate(ebn0)
#     ber_truth = np.array(
#         [
#             0.174463145727110,
#             0.163824306532977,
#             0.153482669976567,
#             0.143468811841223,
#             0.133803341957967,
#             0.124495699481206,
#             0.115543710547806,
#             0.106934252550828,
#             0.098645263673661,
#             0.090649120494965,
#             0.082917115691014,
#             0.075424473431926,
#             0.068155130170213,
#             0.061105459826110,
#             0.054286271091600,
#             0.047722726091144,
#             0.041452236766253,
#             0.035520763591050,
#             0.029978158683620,
#             0.024873215993230,
#             0.020248957901763,
#         ]
#     )
#     assert np.allclose(ber, ber_truth)

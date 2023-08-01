"""
Matlab:
    for M = [2 4 8 16]
        disp(M)
        ebn0 = -10:0.5:10;
        [ber, ser] = berawgn(ebn0, 'psk', M, 'nondiff');
        disp(ber')
    end
"""
import numpy as np

import sdr


def test_bpsk():
    psk = sdr.PSK(2)
    ebn0 = np.arange(-10, 10.5, 0.5)
    ber = psk.ber(ebn0)
    ber_truth = np.array(
        [
            0.327360423009289,
            0.317852409611940,
            0.307910470715079,
            0.297531198428390,
            0.286714527581443,
            0.275464401165253,
            0.263789505256266,
            0.251704067880203,
            0.239228710767672,
            0.226391335744460,
            0.213228018357620,
            0.199783870120343,
            0.186113817483389,
            0.172283230596987,
            0.158368318809598,
            0.144456193925057,
            0.130644488522829,
            0.117040408061573,
            0.103759095953406,
            0.090921205158365,
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
    np.testing.assert_array_almost_equal(ber, ber_truth)


def test_qpsk():
    psk = sdr.PSK(4)
    ebn0 = np.arange(-10, 10.5, 0.5)
    ber = psk.ber(ebn0)
    ber_truth = np.array(
        [
            0.327360423009289,
            0.317852409611940,
            0.307910470715079,
            0.297531198428390,
            0.286714527581443,
            0.275464401165253,
            0.263789505256266,
            0.251704067880203,
            0.239228710767672,
            0.226391335744460,
            0.213228018357620,
            0.199783870120343,
            0.186113817483389,
            0.172283230596987,
            0.158368318809598,
            0.144456193925057,
            0.130644488522829,
            0.117040408061573,
            0.103759095953406,
            0.090921205158365,
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
    np.testing.assert_array_almost_equal(ber, ber_truth)


def test_8psk():
    psk = sdr.PSK(8)
    ebn0 = np.arange(-10, 10.5, 0.5)
    ber = psk.ber(ebn0)
    ber_truth = np.array(
        [
            0.353094253235228,
            0.344335113611685,
            0.335139666224955,
            0.325506463349061,
            0.315438588118110,
            0.304944433236583,
            0.294038478524357,
            0.282742023162907,
            0.271083814932028,
            0.259100505104527,
            0.246836845935333,
            0.234345540679568,
            0.221686657567451,
            0.208926533639903,
            0.196136126436043,
            0.183388824893655,
            0.170757806495827,
            0.158313121836730,
            0.146118789393980,
            0.134230272671424,
            0.122692761078506,
            0.111540652877562,
            0.100798515869953,
            0.090483569561722,
            0.080609413550440,
            0.071190383412915,
            0.062245644984332,
            0.053802047467505,
            0.045894918465536,
            0.038566394957631,
            0.031861441420881,
            0.025822238602581,
            0.020481966282913,
            0.015859067150679,
            0.011952902270821,
            0.008741408289346,
            0.006181056083768,
            0.004209135737169,
            0.002748133589174,
            0.001711694683183,
            0.001011395320989,
        ]
    )
    np.testing.assert_array_almost_equal(ber, ber_truth)


def test_16psk():
    psk = sdr.PSK(16)
    ebn0 = np.arange(-10, 10.5, 0.5)
    ber = psk.ber(ebn0)
    ber_truth = np.array(
        [
            0.378105883434971,
            0.370626881146121,
            0.362765105989891,
            0.354518230113845,
            0.345887371027186,
            0.336877590400389,
            0.327498366960657,
            0.317764016584956,
            0.307694029535573,
            0.297313293315365,
            0.286652170693255,
            0.275746406748512,
            0.264636846472345,
            0.253368954984952,
            0.241992144332456,
            0.230558921760716,
            0.219123881364191,
            0.207742561403029,
            0.196470182397678,
            0.185360268722168,
            0.174463145727110,
            0.163824306532977,
            0.153482669976567,
            0.143468811841223,
            0.133803341957967,
            0.124495699481206,
            0.115543710547806,
            0.106934252550828,
            0.098645263673661,
            0.090649120494965,
            0.082917115691014,
            0.075424473431926,
            0.068155130170213,
            0.061105459826110,
            0.054286271091600,
            0.047722726091144,
            0.041452236766253,
            0.035520763591050,
            0.029978158683620,
            0.024873215993230,
            0.020248957901763,
        ]
    )
    np.testing.assert_array_almost_equal(ber, ber_truth)

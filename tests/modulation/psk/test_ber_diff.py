"""
MATLAB:
    for M = [2 4]
        disp(M)
        ebn0 = -10:0.5:10;
        [ber, ser] = berawgn(ebn0, 'psk', M, 'diff');
        disp(ber')
    end
"""

import numpy as np

import sdr


def test_bpsk():
    psk = sdr.PSK(2)
    ebn0 = np.arange(-10, 10.5, 0.5)
    ber = psk.ber(ebn0, diff_encoded=True)
    ber_truth = np.array(
        [
            0.440391152912936,
            0.433644510631647,
            0.426203225478195,
            0.418012768780312,
            0.409018614510386,
            0.399167529711843,
            0.388409204345841,
            0.376698260185522,
            0.363996669424219,
            0.350276597688599,
            0.335523661089805,
            0.319740550720162,
            0.302950928850298,
            0.285203438104105,
            0.266575588814039,
            0.247177203923486,
            0.227153012282875,
            0.206683901884707,
            0.185986291920676,
            0.165309079221830,
            0.144927686780961,
            0.125134915005245,
            0.106228587716503,
            0.088496395091440,
            0.072198837388899,
            0.057551703083708,
            0.044709972057108,
            0.033755290983041,
            0.024689095178100,
            0.017432958853827,
            0.011836837227531,
            0.007694629619095,
            0.004765173696157,
            0.002795690771783,
            0.001544155578016,
            0.000797274593284,
            0.000381742656596,
            0.000167984966513,
            0.000067252195258,
            0.000024217493303,
            0.000007744186445,
        ]
    )
    np.testing.assert_array_almost_equal(ber, ber_truth)


def test_qpsk():
    psk = sdr.PSK(4)
    ebn0 = np.arange(-10, 10.5, 0.5)
    ber = psk.ber(ebn0, diff_encoded=True)
    ber_truth = np.array(
        [
            0.440391152912936,
            0.433644510631647,
            0.426203225478195,
            0.418012768780312,
            0.409018614510386,
            0.399167529711843,
            0.388409204345841,
            0.376698260185522,
            0.363996669424219,
            0.350276597688599,
            0.335523661089805,
            0.319740550720162,
            0.302950928850298,
            0.285203438104105,
            0.266575588814039,
            0.247177203923486,
            0.227153012282875,
            0.206683901884707,
            0.185986291920676,
            0.165309079221830,
            0.144927686780961,
            0.125134915005245,
            0.106228587716503,
            0.088496395091440,
            0.072198837388899,
            0.057551703083708,
            0.044709972057108,
            0.033755290983041,
            0.024689095178100,
            0.017432958853827,
            0.011836837227531,
            0.007694629619095,
            0.004765173696157,
            0.002795690771783,
            0.001544155578016,
            0.000797274593284,
            0.000381742656596,
            0.000167984966513,
            0.000067252195258,
            0.000024217493303,
            0.000007744186445,
        ]
    )
    np.testing.assert_array_almost_equal(ber, ber_truth)
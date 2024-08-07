"""
MATLAB:
    for M = [2 4 8 16]
        disp(M)
        ebn0 = -10:0.5:10;
        [ber, ser] = berawgn(ebn0, 'psk', M, 'nondiff');
        disp(ser')
    end
"""

import numpy as np

import sdr


def test_bpsk():
    psk = sdr.PSK(2)
    ebn0 = np.arange(-10, 10.5, 0.5)
    esn0 = sdr.ebn0_to_esn0(ebn0, psk.bps)
    ser = psk.ser(esn0)
    ser_truth = np.array(
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
    np.testing.assert_almost_equal(ser, ser_truth)


def test_qpsk():
    psk = sdr.PSK(4)
    ebn0 = np.arange(-10, 10.5, 0.5)
    esn0 = sdr.ebn0_to_esn0(ebn0, psk.bps)
    ser = psk.ser(esn0)
    ser_truth = np.array(
        [
            0.547555999465757,
            0.534674664927764,
            0.521012083454177,
            0.506537582818546,
            0.491223834836636,
            0.475048166021175,
            0.457994107429186,
            0.440053197972964,
            0.421227045479781,
            0.401529634588759,
            0.380989848902523,
            0.359654145480425,
            0.337589281908538,
            0.314884949649039,
            0.291656113216617,
            0.268044795886800,
            0.244220994664267,
            0.220382359003927,
            0.196752241913744,
            0.173575744769280,
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


def test_8psk():
    psk = sdr.PSK(8)
    ebn0 = np.arange(-10, 10.5, 0.5)
    esn0 = sdr.ebn0_to_esn0(ebn0, psk.bps)
    ser = psk.ser(esn0)
    ser_truth = np.array(
        [
            0.726752782740187,
            0.716461451604954,
            0.705438137647736,
            0.693637858501394,
            0.681015417977732,
            0.667526132640204,
            0.653126731724947,
            0.637776447153047,
            0.621438305855023,
            0.604080629555580,
            0.585678737175485,
            0.566216831937906,
            0.545690039309295,
            0.524106543748526,
            0.501489753190243,
            0.477880402207498,
            0.453338490359013,
            0.427944943934869,
            0.401802889234069,
            0.375038434233341,
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
    np.testing.assert_almost_equal(ser, ser_truth)


def test_16psk():
    psk = sdr.PSK(16)
    ebn0 = np.arange(-10, 10.5, 0.5)
    esn0 = sdr.ebn0_to_esn0(ebn0, psk.bps)
    ser = psk.ser(esn0)
    ser_truth = np.array(
        [
            0.845496122720940,
            0.838865953040675,
            0.831737874746586,
            0.824077865256600,
            0.815850902101142,
            0.807021249396889,
            0.797552807916219,
            0.787409527755000,
            0.776555878935490,
            0.764957370340767,
            0.752581101885667,
            0.739396329821041,
            0.725375021515394,
            0.710492375089578,
            0.694727282308978,
            0.678062721501042,
            0.660486081638424,
            0.641989438568998,
            0.622569827459877,
            0.602229577713106,
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
    np.testing.assert_almost_equal(ser, ser_truth)

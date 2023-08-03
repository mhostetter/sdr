"""
Matlab:
    N = [3 5 7 9 11 13 15 17 19 21 23 25]
    R = [2 2 3 5 3  5  7  7  7  11 11 11]
    for ii = 1:length(N)
        seq = zadoffChuSeq(R(ii), N(ii));
        disp(N(ii))
        disp(R(ii))
        disp(seq)
    end
"""
import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(TypeError):
        # N must be an integer
        sdr.zadoff_chu(13.0, 3)
    with pytest.raises(TypeError):
        # R must be an integer
        sdr.zadoff_chu(13, 3.0)
    with pytest.raises(TypeError):
        # Shift must be an integer
        sdr.zadoff_chu(13, 3, 1.0)

    with pytest.raises(ValueError):
        # N must be at least 2
        sdr.zadoff_chu(1, 3)
    with pytest.raises(ValueError):
        # 0 < R < N
        sdr.zadoff_chu(13, -3)
    with pytest.raises(ValueError):
        # 0 < R < N
        sdr.zadoff_chu(13, 25)
    with pytest.raises(ValueError):
        # N and R must be coprime
        sdr.zadoff_chu(15, 3)


def test_3_2():
    seq = sdr.zadoff_chu(3, 2)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.500000000000000 + 0.866025403784438j,
            1.000000000000000 + 0.000000000000000j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_5_2():
    seq = sdr.zadoff_chu(5, 2)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.809016994374947 - 0.587785252292473j,
            0.309016994374948 - 0.951056516295154j,
            -0.809016994374947 - 0.587785252292474j,
            1.000000000000000 + 0.000000000000001j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_7_3():
    seq = sdr.zadoff_chu(7, 3)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.900968867902419 - 0.433883739117558j,
            -0.222520933956314 - 0.974927912181824j,
            -0.900968867902419 + 0.433883739117558j,
            -0.222520933956310 - 0.974927912181825j,
            -0.900968867902415 - 0.433883739117566j,
            1.000000000000000 + 0.000000000000002j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_9_5():
    seq = sdr.zadoff_chu(9, 5)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.939692620785908 + 0.342020143325669j,
            -0.500000000000001 + 0.866025403784438j,
            -0.499999999999997 - 0.866025403784440j,
            -0.939692620785908 + 0.342020143325670j,
            -0.499999999999996 - 0.866025403784441j,
            -0.500000000000013 + 0.866025403784431j,
            -0.939692620785909 + 0.342020143325668j,
            1.000000000000000 + 0.000000000000005j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_11_3():
    seq = sdr.zadoff_chu(11, 3)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.142314838273285 - 0.989821441880933j,
            0.415415013001886 + 0.909631995354519j,
            -0.654860733945286 + 0.755749574354258j,
            -0.142314838273288 + 0.989821441880932j,
            0.841253532831184 - 0.540640817455593j,
            -0.142314838273282 + 0.989821441880933j,
            -0.654860733945286 + 0.755749574354258j,
            0.415415013001885 + 0.909631995354519j,
            -0.142314838273278 - 0.989821441880934j,
            1.000000000000000 + 0.000000000000011j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_13_5():
    seq = sdr.zadoff_chu(13, 5)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.748510748171101 - 0.663122658240795j,
            0.568064746731157 - 0.822983865893656j,
            -0.354604887042534 - 0.935016242685416j,
            0.568064746731156 + 0.822983865893656j,
            0.120536680255321 + 0.992708874098054j,
            0.885456025653215 - 0.464723172043759j,
            0.120536680255313 + 0.992708874098055j,
            0.568064746731157 + 0.822983865893656j,
            -0.354604887042530 - 0.935016242685417j,
            0.568064746731146 - 0.822983865893663j,
            -0.748510748171118 - 0.663122658240776j,
            1.000000000000000 + 0.000000000000022j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_15_7():
    seq = sdr.zadoff_chu(15, 7)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.978147600733806 - 0.207911690817759j,
            -0.809016994374947 - 0.587785252292473j,
            0.309016994374947 + 0.951056516295154j,
            -0.500000000000002 + 0.866025403784437j,
            1.000000000000000 + 0.000000000000002j,
            0.309016994374945 + 0.951056516295154j,
            0.913545457642604 - 0.406736643075793j,
            0.309016994374944 + 0.951056516295155j,
            1.000000000000000 - 0.000000000000002j,
            -0.500000000000028 + 0.866025403784422j,
            0.309016994374954 + 0.951056516295152j,
            -0.809016994374951 - 0.587785252292469j,
            -0.978147600733798 - 0.207911690817795j,
            1.000000000000000 + 0.000000000000033j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_17_7():
    seq = sdr.zadoff_chu(17, 7)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.850217135729614 - 0.526432162877356j,
            0.092268359463302 - 0.995734176295034j,
            -0.982973099683902 - 0.183749517816570j,
            0.739008917220661 - 0.673695643646555j,
            0.445738355776539 - 0.895163291355062j,
            -0.602634636379253 + 0.798017227280242j,
            -0.982973099683903 + 0.183749517816562j,
            0.445738355776541 + 0.895163291355061j,
            -0.982973099683902 + 0.183749517816567j,
            -0.602634636379267 + 0.798017227280231j,
            0.445738355776537 - 0.895163291355063j,
            0.739008917220651 - 0.673695643646566j,
            -0.982973099683900 - 0.183749517816581j,
            0.092268359463338 - 0.995734176295031j,
            -0.850217135729610 - 0.526432162877363j,
            1.000000000000000 + 0.000000000000014j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_19_7():
    seq = sdr.zadoff_chu(19, 7)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.677281571625741 - 0.735723910673132j,
            0.789140509396394 - 0.614212712689668j,
            0.245485487140799 - 0.969400265939330j,
            -0.401695424652972 + 0.915773326655056j,
            -0.986361303402722 + 0.164594590280735j,
            -0.082579345472328 + 0.996584493006670j,
            -0.401695424652966 - 0.915773326655059j,
            -0.082579345472337 - 0.996584493006669j,
            -0.879473751206491 + 0.475947393037070j,
            -0.082579345472314 - 0.996584493006671j,
            -0.401695424652976 - 0.915773326655055j,
            -0.082579345472319 + 0.996584493006671j,
            -0.986361303402727 + 0.164594590280707j,
            -0.401695424652970 + 0.915773326655057j,
            0.245485487140792 - 0.969400265939332j,
            0.789140509396396 - 0.614212712689665j,
            -0.677281571625738 - 0.735723910673134j,
            1.000000000000000 - 0.000000000000006j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_21_11():
    seq = sdr.zadoff_chu(21, 11)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.988830826225129 + 0.149042266176174j,
            -0.900968867902420 + 0.433883739117556j,
            0.623489801858737 - 0.781831482468027j,
            0.074730093586430 - 0.997203797181180j,
            0.623489801858726 + 0.781831482468035j,
            1.000000000000000 + 0.000000000000010j,
            -0.500000000000007 + 0.866025403784434j,
            0.623489801858719 + 0.781831482468042j,
            -0.900968867902425 + 0.433883739117547j,
            0.365341024366377 + 0.930873748644211j,
            -0.900968867902429 + 0.433883739117538j,
            0.623489801858692 + 0.781831482468063j,
            -0.500000000000020 + 0.866025403784427j,
            1.000000000000000 + 0.000000000000049j,
            0.623489801858677 + 0.781831482468075j,
            0.074730093586461 - 0.997203797181177j,
            0.623489801858777 - 0.781831482467995j,
            -0.900968867902475 + 0.433883739117442j,
            -0.988830826225150 + 0.149042266176033j,
            1.000000000000000 + 0.000000000000098j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_23_11():
    seq = sdr.zadoff_chu(23, 11)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.990685946036331 - 0.136166649096247j,
            -0.917211301505452 - 0.398401089846243j,
            0.682553143218652 + 0.730835964278126j,
            0.203456013052629 + 0.979084087682324j,
            0.460065037731157 - 0.887885218402373j,
            0.962917287347803 - 0.269796771157012j,
            -0.775711290704415 - 0.631087944326058j,
            0.203456013052645 - 0.979084087682321j,
            -0.990685946036335 + 0.136166649096215j,
            -0.334879612170962 - 0.942260922118829j,
            -0.917211301505468 + 0.398401089846208j,
            -0.334879612170953 - 0.942260922118833j,
            -0.990685946036338 + 0.136166649096196j,
            0.203456013052687 - 0.979084087682312j,
            -0.775711290704391 - 0.631087944326089j,
            0.962917287347822 - 0.269796771156945j,
            0.460065037731197 - 0.887885218402352j,
            0.203456013052572 + 0.979084087682336j,
            0.682553143218566 + 0.730835964278206j,
            -0.917211301505410 - 0.398401089846342j,
            -0.990685946036311 - 0.136166649096388j,
            1.000000000000000 + 0.000000000000108j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)


def test_25_11():
    seq = sdr.zadoff_chu(25, 11)
    seq_truth = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            -0.929776485888251 - 0.368124552684678j,
            -0.425779291565072 - 0.904827052466020j,
            -0.637423989748691 + 0.770513242775788j,
            -0.809016994374945 - 0.587785252292477j,
            -0.809016994374953 + 0.587785252292466j,
            0.062790519529324 - 0.998026728428271j,
            -0.425779291565060 - 0.904827052466026j,
            0.535826794978986 + 0.844327925502022j,
            0.309016994374936 + 0.951056516295157j,
            0.309016994374973 - 0.951056516295145j,
            0.968583161128635 - 0.248689887164840j,
            -0.425779291565042 - 0.904827052466034j,
            0.968583161128637 - 0.248689887164830j,
            0.309016994374965 - 0.951056516295148j,
            0.309016994374881 + 0.951056516295175j,
            0.535826794978965 + 0.844327925502035j,
            -0.425779291565041 - 0.904827052466034j,
            0.062790519529376 - 0.998026728428268j,
            -0.809016994375018 + 0.587785252292376j,
            -0.809016994374926 - 0.587785252292503j,
            -0.637423989748721 + 0.770513242775763j,
            -0.425779291564954 - 0.904827052466075j,
            -0.929776485888222 - 0.368124552684753j,
            1.000000000000000 + 0.000000000000118j,
        ]
    )
    np.testing.assert_array_almost_equal(seq, seq_truth)

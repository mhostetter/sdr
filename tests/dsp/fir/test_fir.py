import numpy as np
import pytest
import scipy.signal

import sdr


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_non_streaming(mode):
    N = 50
    rng = np.random.default_rng()
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)  # Input signal
    h = rng.standard_normal(10) + 1j * rng.standard_normal(10)  # FIR impulse response

    fir = sdr.FIR(h)
    y = fir(x, mode)
    y_truth = scipy.signal.convolve(x, h, mode=mode)

    assert y.shape == y_truth.shape
    assert np.allclose(y, y_truth)


def test_streaming():
    N = 50
    rng = np.random.default_rng()
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)  # Input signal
    h = rng.standard_normal(10) + 1j * rng.standard_normal(10)  # FIR impulse response

    fir = sdr.FIR(h, streaming=True)

    d = 5  # Stride
    y = np.zeros(N, dtype=complex)
    for i in range(0, N, d):
        y[i : i + d] = fir(x[i : i + d])

    y_truth = scipy.signal.convolve(x, h, mode="full")[0:N]

    assert y.shape == y_truth.shape
    assert np.allclose(y, y_truth)


def test_streaming_match_full():
    N = 50
    rng = np.random.default_rng()
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)  # Input signal
    h = rng.standard_normal(10) + 1j * rng.standard_normal(10)  # FIR impulse response

    fir1 = sdr.FIR(h)
    y_full = fir1(x, mode="full")

    fir2 = sdr.FIR(h, streaming=True)
    d = 10  # Stride
    y_stream = np.zeros_like(y_full)
    for i in range(0, N, d):
        y_stream[i : i + d] = fir2(x[i : i + d])
    y_stream[i + d :] = fir2.flush()

    np.testing.assert_array_almost_equal(y_full, y_stream)


def test_impulse_response():
    rng = np.random.default_rng()
    h_truth = rng.standard_normal(10) + 1j * rng.standard_normal(10)  # FIR impulse response
    fir = sdr.FIR(h_truth)

    h = fir.impulse_response()
    assert h.shape == h_truth.shape
    assert np.allclose(h, h_truth)

    h = fir.impulse_response(20)
    h_truth = np.concatenate((h_truth, [0] * 10))
    assert h.shape == h_truth.shape
    assert np.allclose(h, h_truth)


def test_group_delay():
    """
    MATLAB:
        >> b = fircls1(54,0.3,0.02,0.008);
        >> [phi, w] = grpdelay(b, 1, 30, 'whole');
    """
    b = np.array(
        [
            0.000615962351741,
            -0.006511121375111,
            -0.006183540431315,
            -0.003722363006434,
            0.001572503340765,
            0.006331717630481,
            0.006057839196610,
            -0.000001882825214,
            -0.007714271124356,
            -0.010142338952650,
            -0.003693339523364,
            0.007892419862013,
            0.015062898684317,
            0.009960454375576,
            -0.005808037223313,
            -0.020311637745345,
            -0.019602542557400,
            -0.000117171744191,
            0.025304978475102,
            0.034483175486423,
            0.013232973417498,
            -0.029434340049555,
            -0.061377964476540,
            -0.045857204952108,
            0.032161361913682,
            0.150419734906289,
            0.257272368662233,
            0.300218735368335,
            0.257272368662233,
            0.150419734906289,
            0.032161361913682,
            -0.045857204952108,
            -0.061377964476540,
            -0.029434340049555,
            0.013232973417498,
            0.034483175486423,
            0.025304978475102,
            -0.000117171744191,
            -0.019602542557400,
            -0.020311637745345,
            -0.005808037223313,
            0.009960454375576,
            0.015062898684317,
            0.007892419862013,
            -0.003693339523364,
            -0.010142338952650,
            -0.007714271124356,
            -0.000001882825214,
            0.006057839196610,
            0.006331717630481,
            0.001572503340765,
            -0.003722363006434,
            -0.006183540431315,
            -0.006511121375111,
            0.000615962351741,
        ]
    )
    omega_truth = np.array(
        [
            0,
            0.209439510239320,
            0.418879020478639,
            0.628318530717959,
            0.837758040957278,
            1.047197551196598,
            1.256637061435917,
            1.466076571675237,
            1.675516081914556,
            1.884955592153876,
            2.094395102393195,
            2.303834612632515,
            2.513274122871834,
            2.722713633111154,
            2.932153143350474,
            3.141592653589793,
            3.351032163829113,
            3.560471674068432,
            3.769911184307752,
            3.979350694547071,
            4.188790204786391,
            4.398229715025710,
            4.607669225265029,
            4.817108735504349,
            5.026548245743669,
            5.235987755982989,
            5.445427266222309,
            5.654866776461628,
            5.864306286700947,
            6.073745796940267,
        ]
    )
    phi_truth = np.array(
        [
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
        ]
    )

    fir = sdr.FIR(b)
    f, phi = fir.group_delay(N=omega_truth.size)
    omega = 2 * np.pi * f

    # Convert to 0 to 2*pi
    omega = np.fft.fftshift(omega)
    omega = np.mod(omega, 2 * np.pi)
    phi = np.fft.fftshift(phi)

    assert np.allclose(omega, omega_truth)
    assert np.allclose(phi, phi_truth)

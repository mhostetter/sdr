import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import Literal

import sdr


def test_linear_lagrange_polys():
    """
    MATLAB:
        >> frc = dsp.FarrowRateConverter(PolynomialOrder=1)
        >> getPolynomialCoefficients(frc)
    """
    farrow = sdr.FarrowResampler(1)
    lagrange_polys = farrow.lagrange_polys
    lagrange_polys_truth = np.array(
        [
            [-1, 1],
            [1, 0],
        ]
    ).T
    assert lagrange_polys.shape == lagrange_polys_truth.shape
    assert np.allclose(lagrange_polys, lagrange_polys_truth)


def test_quadratic_lagrange_polys():
    """
    MATLAB:
        >> frc = dsp.FarrowRateConverter(PolynomialOrder=2)
        >> getPolynomialCoefficients(frc)
    """
    farrow = sdr.FarrowResampler(2)
    lagrange_polys = farrow.lagrange_polys
    lagrange_polys_truth = np.array(
        [
            [1 / 2, -1 / 2, 0],
            [-1, 0, 1],
            [1 / 2, 1 / 2, 0],
        ]
    )
    assert lagrange_polys.shape == lagrange_polys_truth.shape
    assert np.allclose(lagrange_polys, lagrange_polys_truth)


def test_cubic_lagrange_polys():
    """
    MATLAB:
        >> frc = dsp.FarrowRateConverter(PolynomialOrder=3)
        >> getPolynomialCoefficients(frc)
    """
    farrow = sdr.FarrowResampler(3)
    lagrange_polys = farrow.lagrange_polys
    lagrange_polys_truth = np.array(
        [
            [-1 / 6, 1 / 2, -1 / 3, 0],
            [1 / 2, -1, -1 / 2, 1],
            [-1 / 2, 1 / 2, 1, 0],
            [1 / 6, 0, -1 / 6, 0],
        ]
    )
    assert lagrange_polys.shape == lagrange_polys_truth.shape
    assert np.allclose(lagrange_polys, lagrange_polys_truth)


def test_quartic_lagrange_polys():
    """
    MATLAB:
        >> frc = dsp.FarrowRateConverter(PolynomialOrder=4)
        >> getPolynomialCoefficients(frc)
    """
    farrow = sdr.FarrowResampler(4)
    lagrange_polys = farrow.lagrange_polys
    lagrange_polys_truth = np.array(
        [
            [1 / 24, -1 / 12, -1 / 24, 1 / 12, 0],
            [-1 / 6, 1 / 6, 2 / 3, -2 / 3, 0],
            [1 / 4, 0, -5 / 4, 0, 1],
            [-1 / 6, -1 / 6, 2 / 3, 2 / 3, 0],
            [1 / 24, 1 / 12, -1 / 24, -1 / 12, 0],
        ]
    )
    assert lagrange_polys.shape == lagrange_polys_truth.shape
    assert np.allclose(lagrange_polys, lagrange_polys_truth)


def test_linear_interpolate():
    """
    MATLAB:
        >> x = transpose(cos(2 * pi / 10 .* (0:40)));
        >> frc = dsp.FarrowRateConverter(InputSampleRate=1, OutputSampleRate=3.1415926, PolynomialOrder=1);
        >> y = frc(x); y
    """
    rate = 3.1415926
    x = np.cos(2 * np.pi / 10 * np.arange(41))
    y_truth = np.array(
        [
            0,
            0.318309891613572,
            0.636619783227144,
            0.954929674840716,
            0.947815886342874,
            0.887024106522329,
            0.826232326701784,
            0.694932373727445,
            0.535777427920659,
            0.376622482113873,
            0.195855640899806,
            -0.000870691072677,
            -0.197597023045160,
            -0.378031289863166,
            -0.537186235669952,
            -0.696341181476738,
            -0.826770443378400,
            -0.887562223198945,
            -0.948354003019489,
            -0.990854217159966,
            -0.930062437339421,
            -0.869270657518877,
            -0.807608186625655,
            -0.648453240818869,
            -0.489298295012083,
            -0.330143349205297,
            -0.138404273089558,
            0.058322058882925,
            0.255048390855407,
            0.424510422771742,
            0.583665368578528,
            0.742820314385314,
            0.844523892381852,
            0.905315672202397,
            0.966107452022942,
            0.973100768156514,
            0.912308988335969,
            0.851517208515424,
            0.761129053717079,
            0.601974107910293,
            0.442819162103507,
            0.277679237251793,
            0.080952905279310,
            -0.115773426693173,
            -0.311834609873532,
            -0.470989555680318,
            -0.630144501487104,
            -0.789299447293890,
            -0.862277341385305,
            -0.923069121205849,
            -0.983860901026394,
            -0.955347319153061,
            -0.894555539332517,
            -0.833763759511972,
            -0.714649920808502,
            -0.555494975001717,
            -0.396340029194930,
            -0.220227869441545,
            -0.023501537469062,
            0.173224794503420,
            0.358313742782108,
            0.517468688588894,
            0.676623634395681,
            0.819239010568212,
            0.880030790388757,
            0.940822570209302,
            0.998385649970154,
            0.937593870149609,
            0.876802090329064,
            0.816010310508520,
            0.668170787899926,
            0.509015842093140,
            0.349860896286354,
            0.162776501631297,
            -0.033949830341185,
            -0.230676162313668,
            -0.404792875690685,
            -0.563947821497471,
            -0.723102767304257,
            -0.836992459571665,
            -0.897784239392209,
            -0.958576019212754,
            -0.980632200966701,
            -0.919840421146156,
            -0.859048641325612,
            -0.780846600798136,
            -0.621691654991350,
            -0.462536709184564,
            -0.302051465793532,
            -0.105325133821049,
            0.091401198151433,
            0.288127530123916,
            0.451272008599261,
            0.610426954406047,
            0.769581900212833,
            0.854745908575117,
            0.915537688395662,
            0.976329468216207,
            0.962878751963249,
            0.902086972142704,
            0.841295192322160,
            0.734367467889560,
            0.575212522082774,
            0.416057576275988,
            0.244600097983284,
            0.047873766010802,
            -0.148852565961681,
            -0.338596195701051,
            -0.497751141507837,
            -0.656906087314623,
            -0.811707577758024,
            -0.872499357578569,
            -0.933291137399114,
            -0.994082917219659,
            -0.945125302959796,
            -0.884333523139252,
            -0.823541743318707,
            -0.687888334980983,
            -0.528733389174197,
            -0.369578443367411,
            -0.187148730173036,
            0.009577601799446,
            0.206303933771929,
            0.385075328609627,
            0.544230274416414,
            0.703385220223200,
            0.829461026761477,
            0.890252806582022,
            0.951044586402567,
        ]
    )
    verify_output_single(1, rate, x, y_truth)
    verify_output_multiple(1, rate, x, y_truth)


# def test_quadratic_interpolate():
#     """
#     MATLAB:
#         >> x = transpose(cos(2 * pi / 10 .* (0:40)));
#         >> frc = dsp.FarrowRateConverter(InputSampleRate=1, OutputSampleRate=3.1415926, PolynomialOrder=2);
#         >> y = frc(x); y
#     """
#     rate = 3.1415926
#     x = np.cos(2 * np.pi / 10 * np.arange(41))
#     y_truth = np.array(
#         [
#             0,
#             -0.108494352257264,
#             -0.115667517415484,
#             -0.021519495474660,
#             0.391492093876194,
#             0.735431338889712,
#             0.958698771958512,
#             0.983633740896390,
#             0.933925110762587,
#             0.852906511924935,
#             0.726294926698372,
#             0.573066721960692,
#             0.407879173364836,
#             0.216688986553699,
#             0.012342558920938,
#             -0.180044524853649,
#             -0.368523824355970,
#             -0.552061699473587,
#             -0.704289605887355,
#             -0.826870587349102,
#             -0.923281135855453,
#             -0.980990434670456,
#             -0.999896002223392,
#             -0.972353671463072,
#             -0.913501371998902,
#             -0.823339103830885,
#             -0.682782926958250,
#             -0.526062151318260,
#             -0.357382031820095,
#             -0.155775963380254,
#             0.045077893350197,
#             0.233972406222474,
#             0.425360364576760,
#             0.599754570364011,
#             0.742838807447414,
#             0.859026803299673,
#             0.944135154949912,
#             0.990542256908803,
#             0.995089312360208,
#             0.958403312269521,
#             0.890407343474986,
#             0.786504969973656,
#             0.638250967289530,
#             0.478037620747230,
#             0.305203529821724,
#             0.095882900135408,
#             -0.101478385692734,
#             -0.286880327662701,
#             -0.479526615037318,
#             -0.644777151494203,
#             -0.778717719247241,
#             -0.887882359586975,
#             -0.961688514381102,
#             -0.996793419483881,
#             -0.987612332736791,
#             -0.941782663315738,
#             -0.864643025190838,
#             -0.744445621278648,
#             -0.592699047692213,
#             -0.428993130247603,
#             -0.242837855603165,
#             -0.037009796819159,
#             0.156858918106673,
#             0.343605739907276,
#             0.531022575737644,
#             0.687129442864164,
#             0.812232048581647,
#             0.913437256211010,
#             0.975941214149024,
#             0.999743922395692,
#             0.977465063353142,
#             0.922491724601724,
#             0.836208417146457,
#             0.701366312655042,
#             0.546127168166297,
#             0.378928679819378,
#             0.181492141313205,
#             -0.020843346568492,
#             -0.211219490592014,
#             -0.401575080177736,
#             -0.579848246677738,
#             -0.726811444473892,
#             -0.845788482398525,
#             -0.935691493171775,
#             -0.986893254253678,
#             -0.997454543587196,
#             -0.964647504209261,
#             -0.900530496127477,
#             -0.804039415635718,
#             -0.657267044102839,
#             -0.498535328711784,
#             -0.327844269462555,
#             -0.121166386951842,
#             0.077676530027545,
#             0.264560103148758,
#             0.456874130687964,
#             0.626003627857601,
#             0.763823156323389,
#             0.876044256552135,
#             0.954645070469273,
#             0.994544634695063,
#             0.991110364013449,
#             0.949159655305148,
#             0.875898977892998,
#             0.762412758057227,
#             0.612147815622037,
#             0.449923529328673,
#             0.269170282454328,
#             0.061860592519077,
#             -0.133489753558001,
#             -0.318207094894841,
#             -0.509502891437960,
#             -0.669488719277231,
#             -0.798164578412653,
#             -0.902999371042476,
#             -0.970297988103502,
#             -0.998895355473181,
#             -0.982095894679470,
#             -0.931001516640803,
#             -0.848597169898287,
#             -0.719766140550137,
#             -0.566008627212638,
#             -0.400291770016964,
#             -0.207391877047851,
#             -0.003574758014909,
#             0.188283017159858,
#             0.377309235214971,
#             0.559461362427724,
#             0.710303520936629,
#         ]
#     )
#     verify_output_single(2, rate, x, y_truth)
#     verify_output_multiple(2, rate, x, y_truth)


def test_cubic_interpolate():
    """
    MATLAB:
        >> x = transpose(cos(2 * pi / 10 .* (0:40)));
        >> frc = dsp.FarrowRateConverter(InputSampleRate=1, OutputSampleRate=3.1415926, PolynomialOrder=3);
        >> y = frc(x); y
    """
    rate = 3.1415926
    x = np.cos(2 * np.pi / 10 * np.arange(41))
    y_truth = np.array(
        [
            0,
            -0.047676392588320,
            -0.063101249092984,
            -0.014023033430338,
            0.266277568896073,
            0.611163170492668,
            0.926050120909111,
            1.029500918324592,
            0.986878373803676,
            0.872418399947425,
            0.734945203469320,
            0.584991854374295,
            0.413431110308494,
            0.225405057414178,
            0.027410488899684,
            -0.171628405814962,
            -0.363405604304107,
            -0.539817322697238,
            -0.696311401406782,
            -0.825788438094717,
            -0.918670751315298,
            -0.977536424756700,
            -0.999964227019318,
            -0.976803585550941,
            -0.917315238029099,
            -0.823851902303805,
            -0.693749467393040,
            -0.536849217383785,
            -0.360140372931888,
            -0.168139363367298,
            0.030947243568754,
            0.228854736974224,
            0.416620842505493,
            0.587852615267498,
            0.737367947817782,
            0.856291537241518,
            0.939482946874534,
            0.987970636065605,
            0.997094331387979,
            0.963081441783240,
            0.893427926787799,
            0.789491572360998,
            0.650339026959141,
            0.487017192345491,
            0.305643221123498,
            0.110324904067745,
            -0.089204414621221,
            -0.285331180048872,
            -0.468509297179932,
            -0.634035808000975,
            -0.776047078082166,
            -0.884035561418569,
            -0.957335414488507,
            -0.995244466453735,
            -0.990984150326391,
            -0.946319665560800,
            -0.866701636067274,
            -0.751714908191349,
            -0.604867293966799,
            -0.435649191058588,
            -0.249486448798607,
            -0.052151306927850,
            0.147171396845386,
            0.340467077987296,
            0.518917556528248,
            0.678213489098496,
            0.812029105084051,
            0.908961912532859,
            0.972169556064207,
            0.999299317828080,
            0.981692281927561,
            0.926576854976632,
            0.837194963960532,
            0.711505046952598,
            0.557487680215189,
            0.382898625322251,
            0.192510839377059,
            -0.006191800640057,
            -0.204658562828918,
            -0.394198824242811,
            -0.567692208751265,
            -0.720232246760884,
            -0.843684966893908,
            -0.931011992491379,
            -0.983926773508626,
            -0.998710939192835,
            -0.969277324284501,
            -0.903911608123745,
            -0.804744820459677,
            -0.669015400443919,
            -0.508353597503485,
            -0.328918906935655,
            -0.134906020271186,
            0.064514791223644,
            0.261476285159487,
            0.446668867883688,
            0.614679842049810,
            0.759938669188966,
            0.872606612430497,
            0.950127203201119,
            0.992548468728753,
            0.993968207295633,
            0.953797875490219,
            0.878382523095148,
            0.768046720901569,
            0.624399380464489,
            0.457618457630863,
            0.273420477592413,
            0.076861618893320,
            -0.122628037410582,
            -0.317329097338974,
            -0.497723797110750,
            -0.659727044624707,
            -0.797179344583565,
            -0.898735443600808,
            -0.966248946569071,
            -0.997976043631579,
            -0.986018929364707,
            -0.935312533637725,
            -0.850048197983851,
            -0.728850343362049,
            -0.577810398813480,
            -0.405435672396498,
            -0.216769079226043,
            -0.018567262655790,
            0.180341911788538,
            0.371549742786915,
            0.547210200124823,
            0.702680404676781,
        ]
    )
    verify_output_single(3, rate, x, y_truth)
    verify_output_multiple(3, rate, x, y_truth)


# def test_quartic_interpolate():
#     """
#     MATLAB:
#         >> x = transpose(cos(2 * pi / 10 .* (0:40)));
#         >> frc = dsp.FarrowRateConverter(InputSampleRate=1, OutputSampleRate=3.1415926, PolynomialOrder=4);
#         >> y = frc(x); y
#     """
#     rate = 3.1415926
#     x = np.cos(2 * np.pi / 10 * np.arange(41))
#     y_truth = np.array(
#         [
#             0,
#             0.020044229454831,
#             0.021507748666758,
#             0.003663764026691,
#             -0.100188332919438,
#             -0.136103459492100,
#             -0.048809749184460,
#             0.268165290578756,
#             0.633850211329309,
#             0.917641400384073,
#             1.018832038561435,
#             0.981742151955230,
#             0.882781734724146,
#             0.754549392205069,
#             0.608727440410384,
#             0.439517199169081,
#             0.252585815865713,
#             0.054664769383362,
#             -0.144483499652377,
#             -0.337631328524518,
#             -0.518454599256390,
#             -0.677948871384442,
#             -0.810057695239949,
#             -0.910471608460704,
#             -0.974483611249359,
#             -0.999648206105946,
#             -0.984507260442448,
#             -0.930514807261266,
#             -0.839876936221480,
#             -0.714735720540518,
#             -0.561456660479204,
#             -0.386589307969405,
#             -0.195372914666907,
#             0.003710237186827,
#             0.201779229706276,
#             0.392238203721131,
#             0.567530698153432,
#             0.719598314701266,
#             0.842941741629764,
#             0.933065905809632,
#             0.985919149059595,
#             0.999378924565940,
#             0.972674698895832,
#             0.907708761286621,
#             0.806906908568494,
#             0.672566176511964,
#             0.512341700939380,
#             0.332368403640312,
#             0.137580976186546,
#             -0.061995741628254,
#             -0.258356391944968,
#             -0.445446702171355,
#             -0.614618171701488,
#             -0.758772227617268,
#             -0.872936378401032,
#             -0.952471792847013,
#             -0.993995015519058,
#             -0.995632131982993,
#             -0.957571086444901,
#             -0.881841463277080,
#             -0.770692817324819,
#             -0.628186112712953,
#             -0.461539442172353,
#             -0.276632596900978,
#             -0.079409319292052,
#             0.120003952861395,
#             0.314089887813786,
#             0.497073766232615,
#             0.659564142398519,
#             0.795347912770943,
#             0.899941081112842,
#             0.968626049837206,
#             0.998685295597375,
#             0.988528238857549,
#             0.939247008910667,
#             0.852993679194189,
#             0.731921650837544,
#             0.581744248277624,
#             0.409210131100160,
#             0.219726371997918,
#             0.021053896310253,
#             -0.177550446347321,
#             -0.369232535188752,
#             -0.546945151980042,
#             -0.702224546460188,
#             -0.829338221114821,
#             -0.923866219678498,
#             -0.981476351398782,
#             -0.999948949343153,
#             -0.978096464587788,
#             -0.917761865831843,
#             -0.821254988717194,
#             -0.690733966993503,
#             -0.533392668880708,
#             -0.355517381185433,
#             -0.162155537023293,
#             0.037292706972616,
#             0.234454164087696,
#             0.423057125774469,
#             0.594895429206464,
#             0.742464133819856,
#             0.860571522357812,
#             0.944633058365515,
#             0.990981266504527,
#             0.997632993645563,
#             0.964374842289593,
#             0.893183870464843,
#             0.786376015208180,
#             0.647273690220126,
#             0.483286826737535,
#             0.300523078880231,
#             0.104120839010642,
#             -0.095441271310917,
#             -0.290537414624779,
#             -0.475376862940934,
#             -0.640767981422413,
#             -0.780156468128588,
#             -0.888856203869589,
#             -0.962173755795620,
#             -0.997110258481438,
#             -0.991950204423086,
#             -0.947410218796544,
#             -0.865590049783784,
#             -0.748672649496563,
#             -0.601688111485436,
#             -0.431585540604024,
#             -0.243960263715000,
#             -0.045819658452908,
#             0.153205943999605,
#         ]
#     )
#     verify_output_single(4, rate, x, y_truth)
#     verify_output_multiple(4, rate, x, y_truth)


def test_linear_decimate():
    """
    MATLAB:
        >> x = transpose(cos(2 * pi / 20 .* (0:40)));
        >> frc = dsp.FarrowRateConverter(InputSampleRate=3.1415926, OutputSampleRate=1, PolynomialOrder=1);
        >> y = frc(x); y
    """
    rate = 1 / 3.1415926
    x = np.cos(2 * np.pi / 20 * np.arange(41))
    y_truth = np.array(
        [
            0,
            0.777692216810960,
            -0.087509039355468,
            -0.869352230009264,
            -0.870609535449398,
            -0.090244395986277,
            0.775733917676395,
            0.951489754224212,
            0.267997831328022,
            -0.648476508285881,
            -0.979643132596578,
            -0.432366763413856,
            0.503907005689578,
            0.992203489031056,
        ]
    )
    verify_output_single(1, rate, x, y_truth)
    verify_output_multiple(1, rate, x, y_truth)


# def test_quadratic_decimate():
#     """
#     MATLAB:
#         >> x = transpose(cos(2 * pi / 20 .* (0:40)));
#         >> frc = dsp.FarrowRateConverter(InputSampleRate=3.1415926, OutputSampleRate=1, PolynomialOrder=2);
#         >> y = frc(x); y
#     """
#     rate = 1 / 3.1415926
#     x = np.cos(2 * np.pi / 20 * np.arange(41))
#     y_truth = np.array(
#         [
#             0,
#             0.935757446057648,
#             0.221507955019479,
#             -0.691434562692190,
#             -0.983711818913181,
#             -0.393554624276109,
#             0.549523025423701,
#             0.999996165064500,
#             0.552522458277187,
#             -0.391219467406183,
#             -0.983303300670362,
#             -0.692773071134951,
#             0.219218724153556,
#             0.934663913777506,
#         ]
#     )
#     verify_output_single(2, rate, x, y_truth)
#     verify_output_multiple(2, rate, x, y_truth)


def test_cubic_decimate():
    """
    MATLAB:
        >> x = transpose(cos(2 * pi / 20 .* (0:40)));
        >> frc = dsp.FarrowRateConverter(InputSampleRate=3.1415926, OutputSampleRate=1, PolynomialOrder=3);
        >> y = frc(x); y
    """
    rate = 1 / 3.1415926
    x = np.cos(2 * np.pi / 20 * np.arange(41))
    y_truth = np.array(
        [
            0,
            0.936280875034163,
            0.223264895106954,
            -0.690045376887961,
            -0.983992960394720,
            -0.394769523724284,
            0.548854296030135,
            0.999989097560472,
            0.553500089130073,
            -0.389657076657610,
            -0.982996025186399,
            -0.694057427863402,
            0.217839139214904,
            0.934304143182736,
        ]
    )
    verify_output_single(3, rate, x, y_truth)
    verify_output_multiple(3, rate, x, y_truth)


# def test_quartic_decimate():
#     """
#     MATLAB:
#         >> x = transpose(cos(2 * pi / 20 .* (0:40)));
#         >> frc = dsp.FarrowRateConverter(InputSampleRate=3.1415926, OutputSampleRate=1, PolynomialOrder=4);
#         >> y = frc(x); y
#     """
#     rate = 1 / 3.1415926
#     x = np.cos(2 * np.pi / 20 * np.arange(41))
#     y_truth = np.array(
#         [
#             0,
#             1.024591389885426,
#             0.513552452822828,
#             -0.432839057857544,
#             -0.990735974516745,
#             -0.659412201386193,
#             0.263752436930116,
#             0.950193778521263,
#             0.783795330315027,
#             -0.086106646884044,
#             -0.878718051920742,
#             -0.882638493725661,
#             -0.094359337555737,
#             0.778610314725163,
#         ]
#     )
#     verify_output_single(4, rate, x, y_truth)
#     verify_output_multiple(4, rate, x, y_truth)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("rate", [1, np.e, np.pi])
@pytest.mark.parametrize("mode", ["rate", "full"])
def test_modes_interpolate(order, rate, mode):
    sps = 10
    span = 4
    x = sdr.root_raised_cosine(0.5, span, sps, norm="power")
    # x = np.random.default_rng().normal(size=100)

    compare_modes(order, x, rate, mode)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("rate", [1, np.e, np.pi])
@pytest.mark.parametrize("mode", ["rate", "full"])
def test_modes_decimate(order, rate, mode):
    sps = 10
    span = 4
    x = sdr.root_raised_cosine(0.5, span, sps, norm="power")
    # x = np.random.default_rng().normal(size=100)

    compare_modes(order, x, 1 / rate, mode)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("rate", [1, np.e, np.pi])
@pytest.mark.parametrize("mode", ["rate", "full"])
def test_modes_interpolate_clock_outputs(order, rate, mode):
    sps = 10
    span = 4
    x = sdr.root_raised_cosine(0.5, span, sps, norm="power")
    # x = np.random.default_rng().normal(size=100)

    compare_modes_clock_outputs(order, x, rate, mode)


def compare_modes(order: int, x: npt.NDArray, rate: float, mode: Literal["rate", "full"], stride: int = 10):
    # Non-streaming
    farrow = sdr.FarrowResampler(order, streaming=False)
    y_ns = farrow(x, rate, mode=mode)

    # Streaming
    farrow = sdr.FarrowResampler(order, streaming=True)
    y = []
    for i in range(0, x.size, stride):
        yi = farrow(x[i : i + stride], rate, mode=mode)
        y.append(yi)
    # y.append(farrow.flush(rate, mode=mode))  # Need to flush the filter state
    y_s = np.concatenate(y)

    if False:
        plt.figure()
        sdr.plot.time_domain(x, label="Input")
        sdr.plot.time_domain(y_ns, sample_rate=rate, marker="o", fillstyle="none", label="Non-streaming")
        sdr.plot.time_domain(y_s, sample_rate=rate, marker=".", label="Streaming")
        plt.title(f"Farrow Fractional Delay (order={order}, rate={rate})")
        plt.show()

    assert np.allclose(y_ns, y_s)


def compare_modes_clock_outputs(
    order: int, x: npt.NDArray, rate: float, mode: Literal["rate", "full"], stride: int = 10
):
    n_outputs = int(x.size * rate / 2)  # Make sure we don't ask for too many outputs

    # Non-streaming
    farrow = sdr.FarrowResampler(order, streaming=False)
    y_ns, _ = farrow.clock_outputs(x, rate, n_outputs, mode=mode)

    # Streaming
    farrow = sdr.FarrowResampler(order, streaming=True)
    y = []
    y_sizes = []
    i = 0
    for _ in range(n_outputs // stride):
        yi, n_inputs = farrow.clock_outputs(x[i:], rate, stride, mode=mode)
        y_sizes.append(yi.size)
        i += n_inputs
        y.append(yi)
    # Final call for the remaining output samples
    remaining = n_outputs - (n_outputs // stride) * stride
    if remaining > 0:
        yi, n_inputs = farrow.clock_outputs(x[i:], rate, remaining, mode=mode)
        y_sizes.append(yi.size)
        y.append(yi)
    y_s = np.concatenate(y)

    if False:
        plt.figure()
        sdr.plot.time_domain(x, label="Input")
        sdr.plot.time_domain(y_ns, sample_rate=rate, marker="o", fillstyle="none", label="Non-streaming")
        sdr.plot.time_domain(y_s, sample_rate=rate, marker=".", label="Streaming")
        plt.title(f"Farrow Fractional Delay (order={order}, rate={rate})")
        plt.show()

    assert y_ns.size == n_outputs
    assert y_s.size == n_outputs
    assert np.all(np.array(y_sizes[:-1]) == stride)
    assert np.allclose(y_ns, y_s)


def debug_plot(x: np.ndarray, y: np.ndarray, y_truth: np.ndarray, offset: float, rate: float):
    plt.figure()
    sdr.plot.time_domain(x, sample_rate=1, marker=".", label="x")
    sdr.plot.time_domain(y_truth, sample_rate=rate, marker="o", label="y_truth")
    sdr.plot.time_domain(y, sample_rate=rate, marker="x", label="y")
    plt.legend()
    plt.show()


def verify_output_single(order: int, rate: float, x: np.ndarray, y_truth: np.ndarray):
    farrow = sdr.FarrowResampler(order, streaming=False)
    y = farrow(x, rate, mode="full")
    # debug_plot(x, y, y_truth, farrow._delay, rate)
    assert np.allclose(y, y_truth)


def verify_output_multiple(order: int, rate: float, x: np.ndarray, y_truth: np.ndarray):
    farrow = sdr.FarrowResampler(order, streaming=True)
    ys = []
    for i in range(0, x.size, 10):
        yi = farrow(x[i : i + 10], rate, mode="full")
        ys.append(yi)
    # ys.append(farrow.flush())  # Need to flush the filter state
    y = np.concatenate(ys)
    # debug_plot(x, y, y_truth, farrow._delay, rate)
    assert np.allclose(y, y_truth)

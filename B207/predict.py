import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import math
import os

def run():
    ConfirmedCases = np.array(
        [1, 1, 2, 2, 5, 5, 5, 6, 6, 8, 8, 8, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 16, 16,
        16, 16, 16, 16, 17, 17, 25, 32, 55, 74, 107, 184, 237, 403, 519, 594, 782, 1147, 1586, 2219, 2978, 3212, 4679,
        6512, 9169, 13663, 20030, 26025, 34898, 46136, 56755, 68837, 86693, 105383, 125013, 143912, 165987, 192301, 224560,
        256792, 289087, 321477, 351354, 382747, 413516, 444731, 480667, 515081, 544183, 571440, 598380, 627205, 652611,
        682626, 715656, 743588, 769684, 799512, 825429, 854288, 887858,
        920185, 950581, 977082, 1000785, 1025362, 1051800, 1081020, 1115946, 1143296, 1167593,
        1191678, 1216209, 1240769, 1268180, 1295019, 1320155, 1339022, 1358293, 1381241, 1401649,
        1428467, 1453214, 1477373, 1495736, 1518126, 1539133, 1561830, 1587596, 1611253, 1632364,
        1652431, 1671104, 1690754, 1709303, 1731625, 1756098, 1779731, 1798718, 1816154, 1837656,
        1857511, 1879150, 1904550, 1925710, 1943626, 1961263, 1979647, 2000757, 2023890, 2048756,
        2073964, 2092912, 2112731, 2136401, 2163465, 2191991, 2223553, 2255823, 2280971, 2313123,
        2350198, 2386074, 2426391, 2472385, 2513731, 2554461, 2595744, 2642174, 2693993, 2750622,
        2801983, 2847664, 2898432, 2941517, 3002171, 3062290, 3124786, 3192841, 3252874, 3311312,
        3370208, 3438244, 3506364, 3582184, 3654445, 3716980, 3777456, 3839546, 3904066, 3974630,
        4043070, 4116393, 4181308, 4236083, 4292934, 4359391, 4431244, 4498701, 4567420, 4623604,
        4669149, 4714678, 4773479, 4827936, 4887293, 4946590, 5000709, 5046463, 5094087, 5142088,
        5198137, 5249451, 5314791, 5361712, 5400904, 5437580, 5482614, 5529973, 5574013, 5622842,
        5665887, 5700119, 5736641, 5777001, 5822167, 5867547, 5914395, 5957126, 5991507, 6026895,
        6068759, 6109773, 6153983, 6204376, 6247464, 6278633, 6302200, 6329593, 6363650, 6399723,
        6447501, 6488563, 6522914, 6557342, 6596849, 6635867, 6681004, 6730288, 6772447, 6810862,
        6862834, 6902696, 6941758, 6988869, 7037151, 7081803, 7119311, 7152546, 7195994, 7235428,
        7281081, 7336043, 7384578, 7420293, 7459742, 7504998, 7556060, 7614653, 7671034, 7725952,
        7771893, 7813735, 7865983, 7925748, 7990636, 8059782, 8116518, 8165858, 8233610, 8295581,
        8358864, 8435164, 8517113, 8599842, 8661982, 8729385, 8806228, 8885632, 8976684, 9075924,
        9165619, 9270467, 9355775, 9482891, 9587499, 9716853, 9844858, 9972308, 10087380, 10207953,
        10348449, 10495075, 10659914, 10840303, 11008064, 11144288, 11307233, 11471155, 11644332,
        11835880, 12034177, 12213451, 12360235, 12534684, 12710198, 12893485, 13005807, 13213995,
        13369528, 13509762, 13670332, 13858551, 14061108, 14284721, 14517506, 14733048, 14914060,
        15108918, 15333410, 15555949, 15787464, 16027441, 16245026, 16432729, 16627550, 16836556,
        17083256, 17322981, 17574950, 17766856, 17954675, 18153724, 18351735, 18581353, 18775557,
        18873203, 19099491, 19255126, 19429760, 19630012, 19863696, 20099363, 20252991, 20553301,
        20762047, 20946329, 21181440, 21436884, 21715174, 22010389, 22271084, 22484332, 22699326,
        22926246, 23156608, 23392315, 23635046, 23836726, 24014508, 24157924, 24334630, 24517866,
        24711684, 24902437, 25073050, 25204112, 25356081, 25503621, 25657566, 25826176, 25992744,
        26135056, 26247053, 26382255, 26497588, 26619229, 26743204, 26877601, 26981588, 27071236,
        27161551, 27257183, 27352360, 27458120, 27557758, 27644880, 27709901, 27764087, 27826806,
        27896924, 27966848, 28046145, 28117670, 28174750, 28230970, 28303233, 28377965, 28455466,
        28532812, 28597387, 28648744, 28706973, 28764033, 28831226, 28899277, 28965728, 29023931,
        29064938, 29109974, 29167616, 29225536, 29288010, 29349533, 29402465, 29440686, 29497352,
        29551309, 29610445, 29670983, 29732612, 29787986, 29821754, 29873347, 29926950, 30013910,
        30081375, 30158696, 30221396, 30264493, 30333922, 30395171, 30462210, 30541255, 30611086,
        30674153, 30709125, 30786804, 30847348, 30922386, 31002264, 31084962, 31151497, 31197877,
        31268107, 31345985, 31421360, 31495649, 31575640, 31628013, 31670031])
    ConfirmedDeaths = np.array([1, 1, 6, 7, 11, 12, 14, 17, 21, 22, 28, 33, 43, 51, 58, 70, 97,
                                134, 194, 266, 372, 475, 596, 788, 1031, 1364, 1783, 2305, 2947, 3571, 4284, 5369, 6653,
                                8165, 9595, 11160, 12773, 14544, 17122, 19277, 21480, 23673, 25801, 27674, 29687, 32121,
                                34726, 36869, 38978, 40958, 42928, 45160, 47676, 50128, 52591, 54777, 56487, 57886, 59394,
                                61621, 64036, 66233, 68138, 69870, 71059, 72438, 74684, 76996, 78923, 80684, 82151, 83127,
                                84170, 85770, 87508, 89296, 90966, 92165, 92983, 94210, 95667, 97167, 98355, 99575, 100661,
                                101298, 101890, 102567, 104029, 105136, 106270, 107224, 107837, 108613, 109594, 110605,
                                111617, 112512, 113156, 113613, 114120, 115033, 115907, 116737, 117566, 118299, 118642,
                                119037, 119851, 120590, 121280, 121908, 122470, 122784, 123192, 123940, 124688, 125218,
                                125844, 126346, 126665, 127048, 127605, 128310, 129026, 129693, 129999, 130324, 130700,
                                131863, 132704, 133716, 134522, 135249, 135721, 136165, 137094, 138067, 139022, 139947,
                                140816, 141282, 141845, 142939, 144162, 145248, 146348, 147261, 147795, 148915, 150256,
                                151683, 152909, 154126, 155220, 155651, 156230, 157570, 158967, 160205, 161437, 162512,
                                163072, 163660, 164680, 166183, 167245, 168585, 169589, 170198, 170689, 171962, 173262,
                                174356, 175447, 176415, 176909, 177383, 178621, 179794, 180898, 181868, 182766, 183218,
                                183760, 184810, 185865, 186895, 187847, 188629, 189073, 189370, 189825, 190976, 191887,
                                193064, 193776, 194204, 194636, 195839, 196789, 197648, 198566, 199282, 199546, 199974,
                                201006, 202061, 202966, 203899, 204671, 204985, 205343, 206203, 207140, 208007, 208871,
                                209547, 209925, 210406, 211096, 212015, 213008, 213981, 214633, 215100, 215494, 216290,
                                217280, 218129, 219063, 219832, 220320, 220798, 221726, 222875, 223763, 224722, 225674,
                                226133, 226666, 227640, 228674, 229678, 230744, 231652, 232147, 232720, 234298, 235420,
                                236569, 237803, 238880, 239451, 240230, 241648, 243084, 244288, 245488, 246834, 247609,
                                248438, 250158, 252094, 254134, 256083, 257699, 258731, 259828, 261957, 264218, 265603,
                                267153, 268512, 269548, 270894, 273441, 276260, 279211, 281888, 284234, 285586, 287200,
                                289828, 293005, 295997, 299413, 301870, 303496, 305152, 308255, 311977, 315443, 318386,
                                321038, 322749, 324688, 328081, 331501, 334411, 335811, 337690, 339120, 341123, 344754,
                                348484, 351932, 354086, 356589, 358030, 360128, 363819, 367749, 371715, 375822, 379140,
                                381139, 383235, 387711, 391740, 395714, 399614, 403042, 404910, 406425, 409157, 413563,
                                417744, 421576, 424919, 426818, 428789, 432825, 436789, 440807, 444423, 447194, 449035,
                                451084, 454558, 458440, 462186, 465812, 468412, 469779, 471389, 474464, 477789, 480972,
                                483888, 486077, 487206, 488160, 489892, 492303, 494815, 497430, 499232, 500448, 501751,
                                504026, 507202, 509571, 511698, 513194, 514280, 515796, 517750, 520246, 522186, 523971,
                                525479, 526172, 526915, 528722, 530296, 531868, 533461, 535044, 535622, 536376, 537549,
                                538750, 540393, 541509, 542307, 542795, 543383, 544243, 545724, 547254, 548396, 549176,
                                549690, 550397, 551273, 552352, 553420, 554361, 555078, 555369, 555877, 556718, 559289,
                                560292, 561164, 561873, 562156, 562623, 563446, 564402, 565289, 566224, 566904, 567217])

    Time1 = np.linspace(0, 376, 376)
    Time2 = np.linspace(0, 453 - 38, 453 - 38)
    Time0 = np.linspace(0, 453, 453)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fx(p, x):
        k1, k2, k3, k4 = p
        return k1 * sigmoid(k2 * x + k3) + k4

    def error(p, x, y):
        return fx(p, x) - y


    def get_r(Np: np.array, Nr: np.array):
        Np_bar = np.mean(Np)
        Nr_bar = np.mean(Nr)
        CovXY = np.sum((Np - Np_bar) * (Nr - Nr_bar))
        CovXX = np.sqrt(np.sum((Np - Np_bar) ** 2))
        CovYY = np.sqrt(np.sum((Nr - Nr_bar) ** 2))
        return CovXY / (CovXX * CovYY)

    res1 = leastsq(error, [1, 1, 1, 1], (Time0, ConfirmedCases))
    res2 = leastsq(error, [1, 1, 1, 1], (Time2, ConfirmedDeaths))
    res3 = leastsq(error,[1,1,1,1],(Time1,ConfirmedCases[0:376]))


    print(res1[0])
    print(res2[0])
    print(res3[0])

    print(get_r(fx(res1[0], Time0), ConfirmedCases))
    print(get_r(fx(res2[0], Time2), ConfirmedDeaths))
    plt.plot(Time2, fx(res2[0], Time2), label="Predict")
    plt.scatter(Time2, ConfirmedDeaths,s=1, alpha=0.5, c='red', label="Real")
    plt.title("ConfirmedDeaths")
    plt.xlabel("Day")
    plt.ylabel("People")
    plt.legend(loc=0, ncol=1)
    plt.savefig(r"img/ConfirmedDeaths.png",dpi=800)
    plt.close()


    plt.plot(Time0, fx(res1[0], Time0), label="Predict")
    plt.scatter(Time0, ConfirmedCases, s=1, alpha=0.5, c='red', label="Real")
    plt.title("ConfirmedCases(1)")
    plt.xlabel("Day")
    plt.ylabel("People")
    plt.legend(loc=0, ncol=1)
    plt.savefig(r"img/ConfirmedCases(1).png", dpi=800)
    plt.close()

    plt.plot(Time0, fx(res3[0], Time0), label="Predict")
    plt.scatter(Time0, ConfirmedCases, s=1, alpha=0.5, c='red', label="Real")
    plt.title("ConfirmedCases(2)")
    plt.xlabel("Day")
    plt.ylabel("People")
    plt.legend(loc=0, ncol=1)
    plt.savefig(r"img/ConfirmedCases(2).png", dpi=800)
    plt.close()
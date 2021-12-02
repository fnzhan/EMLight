import numpy as np
from scipy.integrate import quad
from scipy.special import lpmn
import math
import healpy as hp
from sphere_harmonics import spharmonic_eval
from PIL import Image
import utils

def compute_f2(u):
    return quad(lambda x: np.exp(-1 / (1 - x ** 2)), -1, u + 1e-10)[0] / \
           quad(lambda x: np.exp(-1 / (1 - x ** 2)), -1, 1)[0]


def compute_f3(x, B):
    if x < 0:
        print("x is not in the domain of f3!")
    elif x <= 1 / B:
        return 1
    elif x <= 1:
        return compute_f2(1 - 2 * B / (B - 1) * (x - 1 / B))
    else:
        return 0


# evaluate function b at x with parameter B
# migrated from matlab package "NeedMat" by Minjie Fan
def fun_b(x, B):
    return np.sqrt(compute_f3(x / B, B) - compute_f3(x, B))


# compute spherical needlet coefficients based on spherical harmonic coefficients, lmax and jmax
# migrated from matlab package "NeedMat" by Minjie Fan
def spneedlet(coef, lmax, jmax, B=2.0):
    # beta records spherical needlet coefficients
    beta = dict()

    # b_vector records evaluations of the window function b
    BW = 2.0
    b_vector = np.zeros((jmax + 1, lmax))
    for j in range(jmax + 1):
        for l in range(1, lmax + 1):
            b_vector[j, l - 1] = fun_b(l / BW ** j, BW)

    for j in range(jmax + 1):

        # compute Nside from Nside>=[B^{j+1}]/2
        Nside = 2 ** math.ceil(math.log(math.floor(B ** (j + 1)) / 2, 2))
        # compute the total number of cubature points Npix
        Npix = 12 * Nside ** 2
        # compute cubature weights
        lamb = 4 * np.pi / Npix

        Nring = 4 * Nside - 1
        startpix = np.append(hp.ringinfo(Nside, np.arange(1, Nring + 1))[0], Npix)
        tp = hp.pix2ang(Nside, range(Npix))
        thetas = np.array([tp[0][startpix[i]] for i in range(2 * Nside)])




        pre_legendre = dict()
        for l in range(1, lmax + 1):
            temp_mat = np.zeros((l + 1, len(thetas)))
            norm_term = np.array(
                [(-1) ** m * np.sqrt((l + 0.5) * math.factorial(l - m) / math.factorial(l + m)) for m in range(l + 1)])
            for i in range(len(thetas)):
                temp_mat[:, i] = lpmn(l, l, np.cos(thetas[i]))[0][:, l] * norm_term
            temp_mat2 = (np.fliplr(temp_mat[:, range(len(thetas) - 1)]).T * (-1) ** (l + np.arange(l + 1))).T
            pre_legendre[l] = np.hstack((temp_mat, temp_mat2))

        # compute the minimum and maximum index of l
        l_st = int(np.ceil(B ** (j - 1)))
        l_en = int(min(np.floor(B ** (j + 1)), lmax))

        alm = coef.copy()
        for l in range(l_st, l_en + 1):
            alm[l, range(lmax, l + lmax + 1)] *= b_vector[j, l - 1] * np.sqrt(lamb)

        # implement inverse spherical harmonic transformation
        beta[j] = np.zeros(Npix)

        temp_mat = np.zeros((l_en - l_st + 1, Nring))
        for l in range(l_st, l_en + 1):
            temp_mat[l - l_st, :] = pre_legendre[l][0, :]
        term1 = np.conjugate(alm[range(l_st, l_en + 1), lmax]).dot(temp_mat) / np.sqrt(2 * np.pi)

        temp_mat2 = np.zeros((l_en, Nring)).astype("complex")
        for m in range(1, l_en + 1):
            l_st2 = max(m, l_st)
            temp_mat = np.zeros((l_en - l_st2 + 1, Nring))
            for l in range(l_st2, l_en + 1):
                temp_mat[l - l_st2, :] = pre_legendre[l][m, :]
            temp_mat2[m - 1, :] = alm[range(l_st2, l_en + 1), m + lmax].dot(temp_mat) / np.sqrt(2 * np.pi) * (-1) ** m

        for r in range(Nring):
            for k in range(startpix[r], startpix[r + 1]):
                phi = tp[1][k]
                temp_vec = np.exp(np.arange(1, l_en + 1) * 1j * phi)
                product = temp_vec.dot(temp_mat2[:, r])
                beta[j][k] = term1[r].real + 2 * product.real
                # print (startpix[r], startpix[r + 1], phi, beta[j][k])

    return beta


def spneedlet_pair(jmax, B=2.0):
    # location of cubature points
    pix = dict()
    for j in range(jmax + 1):
        Nside = 2 ** math.ceil(math.log(math.floor(B ** (j + 1)) / 2, 2))
        Npix = 12 * Nside ** 2
        pix[j] = np.vstack((hp.pix2vec(Nside, range(Npix))[i] for i in range(3)))
        # 1, 12,   2, 48,   31 = [2, 3], 3,   16*12 = 768 + 60 =  192 + 60 =252 / 2 + 1 = 127

    pix = np.hstack((pix[j] for j in range(jmax + 1)))
    # 3, 12

    cubature_corr = pix.T.dot(pix)
    cubature_pair = []
    cubature_use = []
    for i in range(pix.shape[1]):
        pair_ind = np.where(cubature_corr[i] + 1 < 1e-10)[0][0]
        cubature_pair.append(pair_ind)
        if pair_ind > i:
            cubature_use.append(i)

    return cubature_pair, cubature_use


def visualize(jmax, B=2.0, SN_coeffs=None):
    # location of cubature points
    pix = dict()
    for j in range(jmax + 1):
        Nside = 2 ** math.ceil(math.log(math.floor(B ** (j + 1)) / 2, 2))
        Npix = 12 * Nside ** 2
        pix[j] = np.vstack((hp.pix2vec(Nside, range(Npix))[i] for i in range(3)))
        # 1, 12,   2, 48,   31 = [2, 3], 3,   16*12 = 768 + 60 =  192 + 60 =252 / 2 + 1 = 127

    pix = np.hstack((pix[j] for j in range(jmax + 1)))
    print ('*****')
    # print (SN_coeffs.shape)
    # print (pix.shape)

    cubature_corr = pix.T.dot(pix)
    cubature_pair = []
    cubature_use = []
    for i in range(pix.shape[1]):
        pair_ind = np.where(cubature_corr[i] + 1 < 1e-10)[0][0]
        cubature_pair.append(pair_ind)
        if pair_ind > i:
            cubature_use.append(i)
    _, ln = pix.shape
    dir = torch.from_numpy(pix).float() # 3, 12
    dir = dir.permute(1, 0).contiguous()
    # print (dir.shape)
    # print (ln)
    dir = dir.view(1, ln * 3)


    band = (math.cos(math.atan(2.0 / 1000 ** 0.5)) - 1.0) / math.log(0.5)
    size = torch.ones((1, ln)) * band

    # color = torch.ones((1, ln, 3)) * 255
    print (SN_coeffs.shape)
    color = torch.from_numpy(SN_coeffs).view(1, 126*2, 3)

    color = color.view(1, ln*3).float()
    env = utils.convert_to_panorama(dir, size, color)
    env = np.squeeze(env[0].detach().numpy())
    env = np.transpose(env, (1, 2, 0))

    env = utils.tonemapping(env) * 255.0
    env = Image.fromarray(env.astype('uint8'))
    env.save('env_coef.jpg')

    return cubature_pair, cubature_use


# evaluate spherical needlets up to jmax at location (theta, phi)
# migrated from matlab package "NeedMat" by Minjie Fan
def spneedlet_eval(theta, phi, jmax, B=2.0):
    # lmax is used in summation formula for evaluating spherical needlets,
    # and is different from lmax in spneedlet_tran
    lmax = int(np.floor(B ** (jmax + 1)))
    coef = np.zeros((lmax + 1, 2 * lmax + 1)).astype("complex") #
    for l in range(1, lmax + 1):
        for m in range(l + 1):
            coef[l, m + lmax] = np.conjugate(spharmonic_eval(l, m, theta, phi))

    return spneedlet(coef, lmax, jmax, B)


# evaluate symmetrized spherical needlets up to jmax on grid specified by (theta, phi)
# SN_matrix: number of grid points * number of symmetrized SN basis
def SNvertex(theta, phi, jmax, B=2.0):
    # evaluate spherical needlets up to jmax on grid specified by (theta, phi)
    # jmax = 1
    # Nside = 1, Npix = 12, psi[0] = [3, 12]
    # Nside = 2, Npix = 48, psi[1] = [3, 48]
    # SN_coef = [12,], [48,0]

    ln,  = theta.shape

    cubature_pair, cubature_use = spneedlet_pair(jmax, B)  # 60, 30

    psi = dict()
    for j in range(jmax + 1):
        Nside = 2 ** math.ceil(math.log(math.floor(B ** (j + 1)) / 2, 2))
        Npix = 12 * Nside ** 2
        psi[j] = np.zeros((ln, Npix))

    for k in range(ln):
        SN_coef = spneedlet_eval(theta[k], phi[k], jmax, B)
        for j in range(jmax + 1):
            psi[j][k] = SN_coef[j]
        print (k)
    SN_temp = np.hstack((psi[j] for j in range(jmax + 1)))


    # calculate the evaluation of symmetrized spherical needlets
    # psi_{jk}(theta,phi)=(\tilde{psi}_{jk}(theta,phi)+\tilde{psi}_{jk'}(theta,phi))/2


    # SN_matrix = ((SN_temp + SN_temp[:, cubature_pair])/2)

    # print ('***', SN_matrix.shape)
    SN_temp1 = (SN_temp)[:, cubature_use]
    SN_temp2 = (SN_temp[:, cubature_pair])[:, cubature_use]
    # SN_temp = (SN_temp1 + SN_temp2)/2
    SH_00 = np.array([spharmonic_eval(0, 0, theta[k], phi[k]).real
                      for k in range(ln)]).reshape(ln, 1)
    # SN_matrix = np.hstack((SH_00, SN_matrix))
    SN_matrix1 = np.hstack((SH_00, SN_temp1))
    SN_matrix2 = np.hstack((SH_00, SN_temp2))
    SN_matrix = np.hstack((SH_00, SN_temp))

    return SN_matrix1, SN_matrix2, SN_matrix


# compute spherical needlet coefficients based on spherical harmonic coefficients
# migrated from matlab package "NeedMat" by Minjie Fan
def spneedlet_tran(coef, lmax, B = 2.0):
    # compute jmax from B^{jmax-1}<lmax
	# jmax is the lowest possible level of sperical needlets,
	# that its summation formula contains spherical harmonics up to lmax
	jmax = math.ceil(math.log(lmax, B))

	return spneedlet(coef, lmax, jmax, B)


# compute spherical needlet coefficients for symmetrized spherical harmonic basis
# C_matrix: number of symmetrized SH basis * number of SN basis
def Ctran_asymm(lmax, B = 2.0):

	jmax = math.ceil(math.log(lmax, B))

	for l in range(0, lmax+1, 2):
		for m in range(-l, l+1):
			# coef is spherical harmonic coefficients of symmetrized spherical harmonic basis with subscripts lm
			coef = np.zeros((lmax+1, 2*lmax+1)).astype("complex")
			if m < 0:
				coef[l, m+lmax], coef[l, -m+lmax] = (-1)**m/np.sqrt(2), 1/np.sqrt(2)
			elif m > 0:
				coef[l, m+lmax], coef[l, -m+lmax] = 1j*(-1)**(m+1)/np.sqrt(2), 1j/np.sqrt(2)
			else:
				coef[l, lmax] = 1
			SN_coef = spneedlet_tran(coef, lmax, B)
			SN_coef = np.hstack((SN_coef[j] for j in range(jmax+1)))
			if l == 0 and m == 0:
				SN_coef = np.insert(SN_coef, 0, 1)
				C_matrix = SN_coef.reshape(1, len(SN_coef))
			else:
				SN_coef = np.insert(SN_coef, 0, 0)
				C_matrix = np.vstack((C_matrix, SN_coef.reshape(1, len(SN_coef))))

	return C_matrix

# compute symmetrized spherical needlet coefficients for symmetrized spherical harmonic basis
# C_matrix: number of symmetrized SH basis * number of symmetrized SN basis
def Ctran(lmax, B = 2.0):

	C_matrix = Ctran_asymm(lmax, B)

	jmax = math.ceil(math.log(lmax, B))
	cubature_pair, cubature_use = spneedlet_pair(jmax, B)

	# coefficients corresponding to \tilde{psi}_{jk}(x) and \tilde{psi}_{jk'}(x) are the same due to symmetry of
	# spherical harmonic function. suppose they are alpha, then coefficient corresponding to psi_{jk}(x) is 2*alpha,
	# since alpha*\tilde{psi}_{jk}(x)+alpha*\tilde{psi}_{jk'}(x)=2*alpha*(\tilde{psi}_{jk}(x)+\tilde{psi}_{jk'}(x))/2
	# =2*alpha*psi_{jk}(x)
	# first column in C_matrix corresponds to first SH basis (constant)
	C_matrix_SN = C_matrix[:, 1:]
	C_matrix_SN = (C_matrix_SN+C_matrix_SN[:, cubature_pair])[:, cubature_use]
	C_matrix = np.hstack((C_matrix[:, 0, None], C_matrix_SN))

	return C_matrix

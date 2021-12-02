import numpy as np
from scipy.special import lpmn
from math import factorial



def shIndex(l, m):
	return l*l+l+m

def shTerms(lmax):
	return (lmax + 1) * (lmax + 1)


def P(l, m, x):
	pmm = 1.0
	if (m > 0):
		somx2 = np.sqrt((1.0 - x) * (1.0 + x))
		fact = 1.0
		for i in range(1, m + 1):
			pmm *= (-fact) * somx2
			fact += 2.0

	if (l == m):
		return pmm * np.ones(x.shape)

	pmmp1 = x * (2.0 * m + 1.0) * pmm

	if (l == m + 1):
		return pmmp1

	pll = np.zeros(x.shape)
	for ll in range(m + 2, l + 1):
		pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
		pmm = pmmp1
		pmmp1 = pll

	return pll

def factorial(x):
	if(x == 0):
		return 1.0
	return x * factorial(x-1)

def K(l, m):
	return np.sqrt(((2 * l + 1) * factorial(l-m))
				   / (4*np.pi*factorial(l+m)))

def SH(l, m, theta, phi):
	sqrt2 = np.sqrt(2.0)
	if(m==0):
		if np.isscalar(phi):
			return K(l, m)*P(l,m,np.cos(theta))
		else:
			return K(l, m)*P(l,m,np.cos(theta))*np.ones(phi.shape)
	elif(m>0):
		return sqrt2*K(l,m)*np.cos(m*phi)*P(l,m,np.cos(theta))
	else:
		return sqrt2*K(l,-m)*np.sin(-m*phi)*P(l,-m,np.cos(theta))

def shEvaluate(theta, phi, lmax):
	if np.isscalar(theta):
		coeffsMatrix = np.zeros((1, 1, shTerms(lmax)))
	else:
		coeffsMatrix = np.zeros((theta.shape[0],phi.shape[0],shTerms(lmax)))

	for l in range(0, lmax+1):
		for m in range(-l, l+1):
			index = l*l+l+m
			coeffsMatrix[:,:,index] = SH(l, m, theta, phi)
	return coeffsMatrix




# evaluate spherical harmonic Phi_{lm} at location (theta, phi), where l is the level and m is the phase
# Y: a complex value of evaluation
def spharmonic_eval(l, m, theta, phi):

	sign_m = np.sign(m)
	m = np.abs(m)

	C = np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m)/factorial(l+m))
	P = lpmn(m, l, np.cos(theta))[0][m, l]
	Y = C*P*np.exp(1j*m*phi)

	if sign_m < 0:
		Y = (-1)**m*np.conjugate(Y)

	return Y

# evaluate symmetrized spherical harmonics up to lmax on grid specified by (theta, phi)
# \tilde{Phi}_{lm}=(-1)^m\sqrt{2}Re(Phi_{lm}) if m<0, Phi_{l0} if m=0, (-1)^m\sqrt{2}Img(Phi_{lm}) if m>0
# SH_matrix: number of grid points * number of symmetrized SH basis
def spharmonic(theta, phi, lmax):

	# L = int((lmax+1)*(lmax+2)/2)  # number of symmetrized SH basis
	L = int((lmax + 1) * (lmax + 1))
	SH_matrix = np.zeros((len(theta), L))

	for i in range(len(theta)):  # vertex
		for l in range(0, lmax+1):  # even level SH
			for m in range(-l, l+1):  # SH phase
				# SH_index = int(l*(l+1)/2+m)
				SH_index = l*l+l+m
				Y_lm = spharmonic_eval(l, m, theta[i], phi[i])
				# Y_lm = SH(l, m, theta[i], phi[i])

				if m < 0:
					SH_matrix[i, SH_index] = (-1)**m*np.sqrt(2)*Y_lm.real
				elif m == 0:
					SH_matrix[i, SH_index] = Y_lm.real
				else:
					SH_matrix[i, SH_index] = (-1)**m*np.sqrt(2)*Y_lm.imag

	return SH_matrix

# generate DWI for a single fiber
# response function is along (theta0, phi0) and is evaluated at (theta, phi)
def myresponse(b, ratio, theta0, phi0, theta, phi):

	D = np.diag(np.array([1.0/ratio, 1.0/ratio, 1]))
	u = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

	# rotation matrix around y-axis
	T_theta = np.array([[np.cos(theta0), 0, -np.sin(theta0)], [0, 1, 0], [np.sin(theta0), 0, np.cos(theta0)]])
	# rotation matrix around z-axis
	T_phi = np.array([[np.cos(phi0), np.sin(phi0), 0], [-np.sin(phi0), np.cos(phi0), 0], [0, 0, 1]])
	# rotation matrix T0 satisfies T0*[sin(theta0)*cos(phi0), sin(theta0)*sin(phi0), cos(theta0)]=[0, 0, 1]
	T0 = T_theta.dot(T_phi)

	# plug in T0*u into single tensor model along z-axis (i.e., D is a diagonal matrix)
	y = np.exp(-b*u.T.dot(T0.T).dot(D).dot(T0).dot(u))

	return y
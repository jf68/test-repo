
# Sweep operator

# This statistical operator can provide us
# with a simple and convenit way of 
# performing ML calculations for
# incomplete data. (Little & Rubin 1987)

# See Chapter 6.5 (Rubin 1987)



# This is specifically used for EM imputation.
# In order to achieve that, we need to obtain
# ML estimates for our multivariate normal 
# parameters in the Expectation step.
# Hence, we FORCE our matrix in the function
# parameters to be symmetric and square.

# James Feng - RI Boston
# September 28, 2016


import numpy as np
import pandas as pd


# BEGIN SWEEP OPERATOR
def swp(k, df):
	"""
	k - kth row and column that is swept
	df - the dataframe that consists of the matrix
		to be swept, this should be a matrix that
		consists of the means in the first row
		and column and the covariance matrix appended

	"""

	G = np.matrix(df)
	n = G.shape[0]


	# Check for square matrix
	if G.shape != (n,n):
		raise ValueError('Not a square matrix')

	# Check for symmetric matrix
	if not np.allclose(G-G.T, 0):
		raise ValueError('Not a symmetrical matrix')

	# Check for a reasonable k
	if (k >= n | k <= 0):
		raise ValueError('Not a valid row/column number')

	H = np.zeros([n,n])

	H[k,k] = -1 / G[k,k]
	gkk = G[k,k]

	for i in range(0,n):
		for j in range(0,n):			

			if (i == j & i != k):
				H[i,j] = G[i,j] - (G[i,k]*G[k,j])/gkk

			if (i != j):
				if (i == k):
					H[i,j] = G[k,j]/gkk
					#print "i = k"
				elif (j == k):
					H[i,j] = G[i,k]/gkk
					#print "j = k"
				else:
					H[i,j] = G[i,j] - (G[i,k]*G[k,j])/gkk

					

			#print H[i,j]
	return H

# END SWEEP OPERATOR


newdata = pd.read_csv('swptest.csv')
#print newdata

means = np.mean(newdata, axis = 0)
#print means
means = np.matrix(means)
#print np.transpose(means)

cov = np.cov(newdata, rowvar=False)
#print cov

A = np.matrix(newdata)
#print A
(a,b) = A.shape


B = np.zeros([b,b])

for i in range(0,b):
	for j in range(0,b):
		temp = np.transpose(A[:,i])
		B[i,j] = np.dot(temp, A[:,j])

B = B/a

# Create a G matrix as in (Rubin 1987, p. 113)

G = np.vstack((means, B))
one = np.array([1])
G0 = np.vstack((one, np.transpose(means)))
G = np.hstack((G0, G))

print G


H = swp(0,G)

print H

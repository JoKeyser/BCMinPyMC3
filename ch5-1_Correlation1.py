# -*- coding: utf-8 -*-
"""
Pearson Correlation model: Inferring a correlation coefficient.
Chapter 5.1, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>

TODO: Not running yet, because of matrix manipulation mysteries in PyMC3/Theano.
"""

import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt

dataset = 1 # choose data set 1 or 2 (where 2 is just the first, twice)

data1 = np.array([[0.8, 102], [1.0,  98], [0.5, 100], [0.9, 105], [0.7, 103],
                  [0.4, 110], [1.2,  99], [1.4,  87], [0.6, 113], [1.1,  89],
                  [1.3,  93]])
if dataset == 1:
    x = data1
elif dataset == 2:
    x = np.vstack((data1, data1))

# from help(tt.stacklists), but doesn't work at all!
#import theano.function
#a, b, c, d = tt.scalars('abcd')
#X = tt.stacklists([[a, b], [c, d]])
#f = theano.function([a, b, c, d], X)

model = pm.Model()
with model:
    # priors
    mu = pm.Normal('mu', mu=0, tau=1/100**2, shape=(2,1))
    lmbda = pm.Gamma('lambda', alpha=0.001, beta=0.001, shape=(2,1))
    r = pm.Uniform('r', lower=-1, upper=1)
    sigma = pm.Deterministic('sigma', tt.sqrt(1/lmbda))

    # Reparameterization
    #FIXME: How to create (and then inverse) a simple 2x2 matrix???
    T = tt.stacklists([[1/lmbda[0]         , r*sigma[0]*sigma[1]],
                       [r*sigma[0]*sigma[1],          1/lmbda[1]]])
#    T = tt.stack([1/lmbda[0]         , r*sigma[0]*sigma[1],
#                  r*sigma[0]*sigma[1],          1/lmbda[1]])
#    TI = tt.invert(T)
#    TI = tt.matrix(T)
    # TODO? Side-step inversion by doing it myself, i.e. 1/det(A)*reshuffle(A)?
    testtau = pm.constant(np.eye(2)) # works...
    pm.det(testtau) # works...

    x = pm.MvNormal('x', mu=0, tau=testtau)

#  # Reparameterization
#  sigma[1] <- 1/sqrt(lambda[1])
#  sigma[2] <- 1/sqrt(lambda[2])
#  T[1,1] <- 1/lambda[1]
#  T[1,2] <- r*sigma[1]*sigma[2]
#  T[2,1] <- r*sigma[1]*sigma[2]
#  T[2,2] <- 1/lambda[2]
#  TI[1:2,1:2] <- inverse(T[1:2,1:2])

    # data come from a Gaussian
#    x = pm.Normal('x', mu=mu, sd=sigma, observed=x)

    # instantiate sampler
    stepFunc = pm.Metropolis() # or try pm.NUTS()

    # draw posterior samples (in 4 parallel running chains)
    Nsample = 1000
    Nchains = 2
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

plotVars = ('mu','sigma')
axs = pm.traceplot(traces, vars=plotVars, combined=False)
# plot joint posterior samples
tstr = 'Joint posterior samples'
post = np.vstack([traces['mu'], traces['sigma']])
post = post.transpose()
df = pd.DataFrame(post, columns=plotVars)
ax = df.plot(kind='scatter', x=plotVars[0], y=plotVars[1], alpha=.1, title=tstr)
# -*- coding: utf-8 -*-
"""
Gaussians model: Inferring a mean and standard deviation.
Chapter 4.1, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>
"""

import pymc3 as pm
import numpy as np
import pandas as pd

x = np.array([1.1, 1.9, 2.3, 1.8])

model = pm.Model()
with model:
    # priors
    mu = pm.Normal('mu', mu=0, sd=100) # or tau=0.001 == 1/100**2
    sigma = pm.Uniform('sigma', lower=.1, upper=10)
    # data come from a Gaussian
    x = pm.Normal('x', mu=mu, sd=sigma, observed=x)
    # instantiate sampler
    stepFunc = pm.Metropolis() # or try pm.NUTS()
    # draw posterior samples (in 4 parallel running chains)
    Nsample = 1000
    Nchains = 4
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

plotVars = ('mu','sigma')
axs = pm.traceplot(traces, vars=plotVars, combined=False)
# plot joint posterior samples
tstr = 'Joint posterior samples'
post = np.vstack([traces['mu'], traces['sigma']])
post = post.transpose()
df = pd.DataFrame(post, columns=plotVars)
ax = df.plot(kind='scatter', x=plotVars[0], y=plotVars[1], alpha=.1, title=tstr)
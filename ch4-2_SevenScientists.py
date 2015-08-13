# -*- coding: utf-8 -*-
"""
SevenScientists model: The seven scientists problem.
Chapter 4.2, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import scipy.stats as ss

xdata = np.array([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])
n = len(xdata)

model = pm.Model()
with model:
    # priors
    mu = pm.Normal('mu', mu=0, tau=0.001)
    lmbda = pm.Gamma('lambda', alpha=0.001, beta=0.001, shape=n)
    sigma = pm.Deterministic('sigma', tt.sqrt(1/lmbda))
    # data come from Gaussians with common mean, but n different precisions
    for ii in range(n):
        pm.Normal('x%d' % ii, mu=mu, tau=lmbda[ii], observed=xdata[ii])
    # instantiate sampler
    stepFunc = pm.HamiltonianMC() # pm.NUTS() isn't working well here..?
    # draw posterior samples (in parallel running chains)
    Nsample = 5000
    Nchains = 4
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

plotVars = ('mu','sigma','lambda')
ax = pm.traceplot(traces, vars=plotVars, combined=False)

#%% separately plot each sigma[i] (sigma consists of Nchains x Nsample x n)
sigma = traces.get_values('sigma', combine=False, burn=1000)
sigma = np.squeeze(sigma)
#sigma = sigma[:,burnin:,:]
axs = plt.subplots(n, 2, squeeze=False, figsize=(10, n*2))
for ii in range(n):
    chains = sigma[:,:,ii].transpose()
    axs[1][ii,0].set_title('sigma[%d] histogram' % ii)
    for cc in range(chains.shape[1]):
        chain = chains[:, cc]
        dnsty = ss.kde.gaussian_kde(chain)
        l = np.min(chain)
        u = np.max(chain)
        x = np.linspace(0, 1, 100) * (u - l) + l
        axs[1][ii,0].plot(x, dnsty(x), alpha=.5)
    axs[1][ii,1].set_title('sigma[%d] traces' % ii)
    axs[1][ii,1].plot(chains, alpha=.5)
    if not ii == n-1:
        axs[1][ii,1].set_xticklabels([])
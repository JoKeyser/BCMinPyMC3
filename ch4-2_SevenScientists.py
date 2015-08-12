# -*- coding: utf-8 -*-
"""
SevenScientists model: The seven scientists problem.
Chapter 4.2, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>

TODO: Get the model right, and plot it in a nice way.
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
#import seaborn as sns
#sns.set_style("whitegrid")

xdata = np.array([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])
n = len(xdata)

model = pm.Model() #FIXME
with model:
    # priors
    mu = pm.Normal('mu', mu=0, tau=0.001)
    lamda = pm.Gamma('lambda', alpha=0.001, beta=0.001, shape=n)
    sigma = pm.Deterministic('sigma', 1/tt.sqrt(lamda))
    # data come from Gaussians with common mean, but n different precisions
    x = [[],[],[],[],[],[],[]] #FIXME
    for ii in range(n):
        x[ii] = pm.Normal('x%d' % ii, mu=mu, sd=sigma[ii], observed=xdata[ii])
    # instantiate sampler
    stepFunc = pm.Metropolis() # or try pm.NUTS()
    # draw posterior samples (in 4 parallel running chains)
    Nsample = 1000
    Nchains = 4
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

plotVars = ('mu','sigma')
ax = pm.traceplot(traces, vars=plotVars, combined=False)

#FIXME:
# separately plot each entry in "vector" sigma (which is Nchains x Nsample x n)
sigma = traces.get_values('sigma', combine=False)
sigma = np.squeeze(sigma)
axs = plt.subplots(n, 2, squeeze=False, figsize=(10, n*1.5))
for ii in range(n):
    chains = sigma[:,:,ii].transpose()
    axs[1][ii,0].set_title('sigma[%d] histogram' % ii)
    axs[1][ii,1].set_title('sigma[%d] traces' % ii)
    plt.plot(chains, alpha=.5)
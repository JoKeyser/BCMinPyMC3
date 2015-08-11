# -*- coding: utf-8 -*-
"""
Rate_4 model: Prior and posterior prediction.
Chapter 3.4 Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>
"""

import pymc3 as pm

# default values
#k = 1
#n = 15
# Uncomment for Trompetter Data
k =  24
n = 121

model = pm.Model()

with model:
    # Prior on rate Theta
    theta = pm.Beta('theta', alpha=1, beta=1)
    # Observed Data
    k = pm.Binomial('k', p=theta, n=n, observed=k)
    # Posterior Predictive
    postPredK = pm.Binomial('postPredK', p=theta, n=n)
    # Prior Predictive
    thetaPrior = pm.Beta('thetaPrior', alpha=1, beta=1)
    priorPredK = pm.Binomial('priorPredK', p=thetaPrior, n=n) 
    # instantiate Metropolis-Hastings sampler
    stepFunc = pm.Metropolis()
    # draw 5,000 posterior samples (in 4 parallel running chains)
    Nsample = 5000
    Nchains = 4
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

plotVars = ['theta','priorPredK','thetaPrior','postPredK']
axs = pm.traceplot(traces, vars=plotVars, combined=False)
axs[0][0].set_xlim([0,1]) # manually set x-limits for comparisons
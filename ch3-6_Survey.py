# -*- coding: utf-8 -*-
"""
Survey model: Inferring Return Rate and Number of Surveys from Observed Return.
Chapter 3.6, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>

TODO: Partially works (often, not always!); need to read more about how to use
      pm.Categorical() and pm.ElemwiseCategoricalStep()
"""

import pymc3 as pm
import numpy as np

nmax = 500
k    = (16, 18, 22, 25, 27)
m    = len(k)

model = pm.Model()

# The model in JAGS:
#model{
#  # Observed Returns
#  for (i in 1:m){
#     k[i] ~ dbin(theta,n)
#  }   
#  # Priors on Rate Theta and Number n
#  theta ~ dbeta(1,1)
#  n ~ dcat(p[])
#  for (i in 1:nmax){
#     p[i] <- 1/nmax
#  }
#}   
with model:
    # Priors on rate theta and number n
    theta = pm.Beta('theta', alpha=1, beta=1)
    p = pm.constant(np.ones(nmax)/nmax)
    n = pm.Categorical('n', p=p, shape=1) #FIXME: How to use this properly?
    # Observed Returns
#    k = pm.Binomial('k', p=theta, n=n, observed=k, shape=m)
    # instantiate samplers
    values_np = np.ones(nmax)/nmax #FIXME: How to use this properly?
    step1 = pm.Metropolis([theta])
    step2 = pm.ElemwiseCategoricalStep(var=n, values=values_np)
    stepFunc = [step1, step2]
    # draw posterior samples (in 4 parallel running chains), TODO: very slow!?
    Nsample = 100
    Nchains = 4
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

plotVars = ['theta','n']
axs = pm.traceplot(traces, vars=plotVars, combined=False)
axs[0][0].set_xlim([0,1]) # manually set x-limits for comparisons

# -*- coding: utf-8 -*-
"""
Reimplementation of Rate_1 model from Chapter 3, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>
"""

import pymc3 as pm

n = 100 # n = 10, or n = 100, or k = 0
k =  50 # k =  5, or k =  99, or n = 1
# For n=100, k=50, theta's posterior gets more narrow, b/c more data.
# For n=100, k=99, theta's posterior narrows down further, and gets almost
#                  squeezed against its maximum, where it drops abruptly.
# For n=1, k=0, theta's posterior is still very broad, but clearly leans toward
#               lower values. Even one data point can have a lot of influence.

model = pm.Model()

with model:
    # Prior Distribution for Rate Theta
    theta = pm.Beta('theta', alpha=1, beta=1)
    # Observed Counts
    k = pm.Binomial('k', p=theta, n=n, observed=k)
    # instantiate Metropolis-Hastings sampler
    stepFunc = pm.Metropolis()
    # draw 20,000 posterior samples (in 2 parallel chains)
    Nsample = 10000
    traces = pm.sample(Nsample, step=stepFunc, njobs=2)

axs = pm.traceplot(traces, vars=['theta'], combined=False)
axs[0][0].set_xlim([0,1]) # manually set x-limits for comparisons
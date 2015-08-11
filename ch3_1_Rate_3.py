# -*- coding: utf-8 -*-
"""
Rate_3 model: Inferring a common rate.
Chapter 3.3, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>
"""

import pymc3 as pm

# default
k1, n1 = 5, 10
k2, n2 = 7, 10
## Exercise 3.3.1
#k1, n1 = 14, 20
#k2, n2 = 16, 20
## Exercise 3.3.2
#k1, n1 =  0, 10
#k2, n2 = 10, 10
## Exercise 3.3.3
#k1, n1 =  5, 10
#k2, n2 =  5, 10

model = pm.Model()

with model:
    # Prior on single rate
    theta = pm.Beta('theta', alpha=1, beta=1)
    # Observed Counts
    k1 = pm.Binomial('k1', p=theta, n=n1, observed=k1)
    k2 = pm.Binomial('k2', p=theta, n=n2, observed=k2)
    # instantiate Metropolis-Hastings sampler
    stepFunc = pm.Metropolis()
    # draw 5,000 posterior samples (in 4 parallel running chains)
    Nsample = 5000
    Nchains = 4
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

axs = pm.traceplot(traces, vars=['theta'], combined=False)
axs[0][0].set_xlim([0,1]) # manually set x-limits for comparisons
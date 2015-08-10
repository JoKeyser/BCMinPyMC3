# -*- coding: utf-8 -*-
"""
Rate_2 model: Difference between two rates.
Chapter 3.2, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>
"""

import pymc3 as pm
import numpy as np
from scipy.stats import mstats

k1 = 5
k2 = 7
n1 = 10
n2 = 10

model = pm.Model()

with model:
    # Prior on Rates
    theta1 = pm.Beta('theta1', alpha=1, beta=1)
    theta2 = pm.Beta('theta2', alpha=1, beta=1)
    # Observed Counts
    k1 = pm.Binomial('k1', p=theta1, n=n1, observed=k1)
    k2 = pm.Binomial('k2', p=theta2, n=n2, observed=k2)
    # Difference between the two rates
    delta = pm.Deterministic('delta', theta1 - theta2)
    # instantiate Metropolis-Hastings sampler
    stepFunc = pm.Metropolis()
    # draw 5,000 posterior samples (in 4 parallel chains)
    Nsample = 5000, Nchains = 4
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

axs = pm.traceplot(traces, vars=['theta1','theta2','delta'], combined=False)
axs[0][0].set_xlim([0,1]) # manually set x-limits for comparisons

# mean of delta:
np.mean(traces['delta'])
# median of delta:
np.median(traces['delta'])
# mode of delta
mstats.mode(traces['delta']) #FIXME! apply to SMOOTHED histogram
# 95% credible interval for delta:
mstats.mquantiles(traces['delta'], (0.025, 0.975))
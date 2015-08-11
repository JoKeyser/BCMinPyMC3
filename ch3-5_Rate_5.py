# -*- coding: utf-8 -*-
"""
Rate_5 model: Posterior prediction.
Chapter 3.3, Bayesian Cognitive Modeling.

Created Aug/2015 by Johannes Keyser <j.keyser@donders.ru.nl>
"""

import pymc3 as pm
import pandas as pd
import numpy as np

# default
k1, n1 =  0, 10
k2, n2 = 10, 10

model = pm.Model()

with model:
    # Prior on single rate
    theta = pm.Beta('theta', alpha=1, beta=1)
    # Observed Counts
    k1 = pm.Binomial('k1', p=theta, n=n1, observed=k1)
    k2 = pm.Binomial('k2', p=theta, n=n2, observed=k2)
    # Posterior Predictive
    postPredK1 = pm.Binomial('postPredK1', p=theta, n=n1)
    postPredK2 = pm.Binomial('postPredK2', p=theta, n=n2)
    # instantiate Metropolis-Hastings sampler
    stepFunc = pm.Metropolis()
    # draw 5,000 posterior samples (in 4 parallel running chains)
    Nsample = 5000
    Nchains = 4
    traces = pm.sample(Nsample, step=stepFunc, njobs=Nchains)

plotVars = ['theta','postPredK1','postPredK2']
axs = pm.traceplot(traces, vars=plotVars, combined=False)
axs[0][0].set_xlim([0,1]) # manually set x-limits for comparisons

#FIXME: recreate the nice density plot from figure 3.11
#FIXME: add actual data point (0,10) into plot
pred = np.array([traces['postPredK1'].flatten(1), traces['postPredK2']])
pred = pred.transpose()
labels = ('Success Count 1', 'Success Count 2')
df = pd.DataFrame(pred, columns=labels)
ax = df.plot(kind='hexbin', x=labels[0], y=labels[1], gridsize=10)
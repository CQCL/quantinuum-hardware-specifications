# # Copyright 2024 Quantinuum (www.quantinuum.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' Functions for analyzing bright state decay data from Quantinuum. '''

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import  erf

from .loading_functions import load_data


def decay_analysis(data_dir: str, 
                   machine: str, 
                   date: str, 
                   decay_type: str):
    ''' Analyze bright decay data and return DataFrame of results. '''

    data = load_data(data_dir, machine, date, decay_type)

    fit_info = {}
    boot_info = {}
    for q in data['survival']:
        xvals = list(data['survival'][q].keys())
        yvals = [
            val/data['shots']
            for val in data['survival'][q].values()
        ]
        fit_info[q] = decay_fit(
            xvals, 
            yvals,
            decay_type=decay_type
        )
        boot_info[q] = bootstrap(
            data['survival'][q], 
            data['shots'], 
            decay_type=decay_type
        )
    return fit_info, boot_info


def decay_analysis_combined(data_dir: str,
                            machine: str, 
                            date: str, 
                            decay_type: str):
    ''' Analyze RB data combined for all qubits. '''

    data = load_data(data_dir, machine, date, decay_type)

    fit_info = {}
    boot_info = {}
    qlist = list(data['survival'].keys())
    xvals = list(data['survival'][qlist[0]])
    ylist = {}
    for x in xvals:
        ylist[x] = 0
        for q in qlist:
            ylist[x] += data['survival'][q][x]

    yvals = [
        val/data['shots']/len(qlist)
        for val in ylist.values()
    ]
    fit_info = decay_fit(
        xvals, 
        yvals,
        decay_type
    )
    boot_info = bootstrap(
        ylist, 
        data['shots']*len(qlist), 
        decay_type
    )
    return fit_info[1], (boot_info['Avg. fidelity upper'] - boot_info['Avg. fidelity lower'])/2


def decay_fit(xvals: list,
              yvals: list,
              decay_type: str):
    ''' Fit data to theoretical bright state population to get rate. '''

    if decay_type == 'Measurement_crosstalk':
        func = measurement_crosstalk
    elif decay_type == 'Reset_crosstalk':
        func = reset_crosstalk

    fit = curve_fit(
        func,
        xvals,
        yvals,
        p0=[0.02, 1e-5],
        bounds = [
            (0, 0),
            (1, 1)
        ]
    )
    metrics = convert_params(fit[0], decay_type)

    return metrics


def convert_params(fit_param, decay_type):
    ''' Convert to standard metrics. '''

    if decay_type == 'Measurement_crosstalk':
        out = [
            fit_param[0],
            5*fit_param[1]/6
        ]
    elif decay_type == 'Reset_crosstalk':
        out = [
            fit_param[0],
            5*fit_param[1]/3
        ]
    return out


def convert_metrics(metrics_param, decay_type):
    ''' Convert to estimated fits. '''
    
    if decay_type == 'Measurement_crosstalk':
        out = [
            metrics_param[0],
            6*metrics_param[1]/5
        ]
    elif decay_type == 'Reset_crosstalk':
        out = [
            metrics_param[0],
            3*metrics_param[1]/5
        ]

    return out

def measurement_crosstalk(m: int, spam:float, gamma: float):
    ''' Decay function for bright state population with measurement errors. '''

    return (1/3)*(1 + spam + np.exp(-3*gamma*m)*(2 - 4*spam))

def reset_crosstalk(m: int, spam: float, gamma: float):
    ''' Decay function for bright state population with reset errors. '''

    return spam + (1/3)*np.exp(-5*gamma*m)*(2 + np.exp(3*gamma*m))*(1 - 2*spam)


def bootstrap(survival: dict,
              shots: int, 
              decay_type: str,
              resamples: int = 1000):
    ''' Parametric bootstrap resample for bright state decay. '''

    xvals = [int(m) for m in survival]
    yvals = [val/shots for val in survival.values()]

    resample = np.random.binomial(
            shots, 
            yvals,
            size=[resamples, len(yvals)]
    )/shots
    boot_sample = np.array([
        decay_fit(xvals, resample[r, :], decay_type)
        for r in range(resamples)
    ])
    thresh = 1/2 + erf(1/np.sqrt(2))/2
    uncertainty ={
        'SPAM lower': 2*np.mean(boot_sample[:,0]) - np.quantile(boot_sample[:,0], thresh),
        'SPAM upper': 2*np.mean(boot_sample[:,0]) - np.quantile(boot_sample[:,0], 1-thresh),
        'Avg. fidelity lower': 2*np.mean(boot_sample[:,1]) - np.quantile(boot_sample[:,1], thresh),
        'Avg. fidelity upper': 2*np.mean(boot_sample[:,1]) - np.quantile(boot_sample[:,1], 1-thresh)
    }
    return uncertainty
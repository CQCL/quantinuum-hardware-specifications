# Copyright 2025 Quantinuum (www.quantinuum.com)
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

''' Functions for plotting bright state decay data from Quantinuum. '''

import pandas as pd
import numpy as np
from qtm_spec.util import avg_uncertainty
import matplotlib.pyplot as plt

from .decay_analysis_functions import (
    measurement_crosstalk, reset_crosstalk, convert_metrics
)
from .loading_functions import load_data
from .zone_names import *


def errorbar_plot(fid_info: dict,
                  data_dir: str,
                  machine: str,
                  date: str,
                  decay_type: str,
                  log_scale=False,
                  savename=None):
    ''' Plot bright state population and fitting. '''

    data = load_data(data_dir, machine, date, decay_type)
    
    if len(fid_info) > 10:
        cmap = plt.cm.turbo  # define the colormap
        color_list = [cmap(i) for i in range(0, cmap.N, cmap.N//len(fid_info))]
    else:
        color_list = [plt.get_cmap("tab10").colors[i] for i in range(10)]

    fig, ax = plt.subplots()
    legend = []
    c = 0
    for q, surv in data['survival'].items():
        xvals = [int(l) for l in data['survival'][q]]
        xrange = np.arange(np.min(xvals), np.max(xvals)+1)

        fit = convert_metrics(fid_info[q], decay_type)
        if decay_type == 'Measurement_crosstalk':
            survival_fit = measurement_crosstalk(xrange, *fit)
        elif decay_type == 'Reset_crosstalk':
            survival_fit = reset_crosstalk(xrange, *fit)
        ax.plot(xrange, survival_fit, "-", color=color_list[c])

        for length in xvals:
            surv_freq = surv[str(length)]/data['shots']
            ax.errorbar(
                length,
                surv_freq,
                yerr=np.sqrt(surv_freq*(1 - surv_freq)/data['shots']),
                fmt="o",
                markersize=5,
                capsize=3,
                ecolor=color_list[c],
                markerfacecolor=[1, 1, 1],
                markeredgecolor=color_list[c],
            )
        legend.append(str(q))
        c += 1

    ax.grid(True, axis="both", linestyle="--")
    ax.set_xlabel("Sequence length (number of measurements)")
    ax.set_ylabel("Success counts")

    if machine == 'H1-1' or (machine == 'H1-2' and int(date.split('_')[0]) > 2022) or machine == 'REIMEI':
        try:
            legend = [zone_labels_1[key] for key in legend]
        except KeyError:
            pass
    elif machine == 'H1-2':
        try:
            legend = [zone_labels_2[key] for key in legend]
        except KeyError:
            pass
    elif machine == 'H2-1':
        try:
            legend = [zone_labels_3[key] for key in legend]
        except KeyError:
            pass
    ax.legend(legend)
    if log_scale:
        ax.set_xscale("log")

    if savename:
        fig.savefig(savename + '.pdf', format='pdf')


def report(fid_info: dict, 
           boot_info: dict,
           machine: str,
           date: str, 
           decay_type: str):
    ''' Returns DataFrame containing summary of results. '''

    if machine == 'H1-1' or (machine == 'H1-2' and int(date.split('_')[0]) > 2022) or machine == 'REIMEI':
        try:
            fid_info = {zone_labels_1[key]: fid for key, fid in fid_info.items()}
            boot_info = {zone_labels_1[key]: fid for key, fid in boot_info.items()}
        except KeyError:
            pass
    elif machine == 'H1-2':
        try:
            fid_info = {zone_labels_2[key]: fid for key, fid in fid_info.items()}
            boot_info = {zone_labels_2[key]: fid for key, fid in boot_info.items()}
        except KeyError:
            pass
    elif machine == 'H2-1':
        try:
            fid_info = {zone_labels_3[key]: fid for key, fid in fid_info.items()}
            boot_info = {zone_labels_3[key]: fid for key, fid in boot_info.items()}
        except KeyError:
            pass

    df1 = pd.DataFrame.from_dict(fid_info).transpose()
    df1.rename(columns={0: 'Decay intercept', 1: 'Avg. infidelity'}, inplace=True)
    boot_new = {
        qname: {
            'Decay intercept uncertainty': (qvals['SPAM upper'] - qvals['SPAM lower'])/2,
            'Avg. infidelity uncertainty': (qvals['Avg. fidelity upper'] - qvals['Avg. fidelity lower'])/2
        }
        for qname, qvals in boot_info.items()
    }
    df2 = pd.DataFrame.from_dict(boot_new).transpose()
    
    result = pd.concat([df1, df2], axis=1).reindex(df1.index)
    result.rename(columns={result.columns[0]: 'Qubits'})
    result = result[['Avg. infidelity', 'Avg. infidelity uncertainty', 'Decay intercept', 'Decay intercept uncertainty']]
    result.loc['Mean'] = result.mean()

    # change uncertainties to geometric means
    result['Decay intercept uncertainty']['Mean'] = avg_uncertainty(
        result['Decay intercept uncertainty'].head(len(result['Decay intercept uncertainty']) - 1).to_list()
    )
    result['Avg. infidelity uncertainty']['Mean'] = avg_uncertainty(
        result['Avg. infidelity uncertainty'].head(len(result['Avg. infidelity uncertainty']) - 1).to_list()
    )
    pd.set_option('display.float_format', lambda x: '%.3E' % x)

    return result
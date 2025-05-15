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

import numpy as np
import pandas as pd
from uncertainties import ufloat

from qtm_spec.decay_analysis_functions import decay_analysis_combined

from qtm_spec.rb_analysis_functions import rb_analysis_combined
from qtm_spec.spam_reporting_functions import spam_combined


def combined_report(data_dir: str, machine: str, date: str, test_list: list):
    ''' Make table of estimates from all methods. '''

    renamed = {
        'SQ_RB': 'Single-qubit gate error',
        'TQ_RB': 'Two-qubit gate error',
        'SQ_RB_legacy': 'Single-qubit gate error (legacy)',
        'TQ_RB_legacy': 'Two-qubit gate error (legacy)',
        'SQ_RB_leakage': 'Single-qubit leakage',
        'TQ_RB_leakage': 'Two-qubit leakage',
        'Memory_RB_legacy': 'Memory error',
        'Measurement_crosstalk': 'Measurement crosstalk error',
        'SPAM': 'SPAM error'
    }
    df_raw = extract_parameters(data_dir, machine, date, test_list)

    df = {}
    for old_name, new_name in renamed.items():
        if old_name in df_raw:
            df[new_name] = ['{:.1uePS}'.format(ufloat(df_raw[old_name][0], df_raw[old_name][1]))]
        else:
            df[new_name] = None
    result = pd.DataFrame.from_dict(df).transpose()
    result.rename(columns={0: 'Error magnitude'}, inplace=True)
    # pd.set_option('display.float_format', lambda x: '%.3E' % x)

    return result


def emulator_parameters(data_dir: str, machine: str, date: str):
    ''' Make dictionary of emulator parameters from all tests. '''

    renamed = {
        'SQ_RB': 'p1',
        'SQ_RB_leakage': 'p1_emission_ratio',
        'TQ_RB': 'p2',
        'TQ_RB_leakage': 'p2_emission_ratio',
        'Memory_RB_legacy': 'dephasing_error',
        'Measurement_crosstalk': 'p_crosstalk_meas',
        'Reset_crosstalk': 'p_crosstalk_init',
        'SPAM0': 'p0_meas',
        'SPAM1': 'p1_meas'
    }
    df_raw = extract_parameters(data_dir, machine, date)

    df = {}
    for old_name, new_name in renamed.items():
        if old_name == 'SQ_RB_leakage':
            if old_name in df_raw:
                df[new_name] = first_sig_fig(2*df_raw[old_name][0]/df_raw['SQ_RB'][0], 2*df_raw[old_name][1]/df_raw['SQ_RB'][0])
            else:
                df[new_name] = None
        elif old_name == 'TQ_RB_leakage':
            if old_name in df_raw:
                df[new_name] = first_sig_fig(df_raw[old_name][0]/df_raw['TQ_RB'][0]/0.543, df_raw[old_name][1]/df_raw['TQ_RB'][0]/0.543)
            else:
                df[new_name] = None
        elif old_name == 'SQ_RB' and 'SQ_RB' not in df_raw:
            df[new_name] = first_sig_fig(df_raw['SQ_RB_legacy'][0], df_raw['SQ_RB_legacy'][1])
        elif old_name == 'TQ_RB' and 'TQ_RB' not in df_raw:
            df[new_name] = first_sig_fig(df_raw['TQ_RB_legacy'][0], df_raw['TQ_RB_legacy'][1])
        else:
            if old_name in df_raw:
                df[new_name] = first_sig_fig(df_raw[old_name][0], df_raw[old_name][1])
            else:
                df[new_name] = None
    
    return df


def extract_parameters(data_dir: str, machine: str, date: str, test_list: list = None):
    ''' Extract parameters from all tests combined over gate zones. '''

    if test_list is None:
        test_list = [
            'SQ_RB',
            'TQ_RB', 
            'Memory_RB', 
            'Measurement_crosstalk', 
            'SPAM'
        ]

    df = {}
    for test in test_list:
        if 'RB' in test:
            val, unc = rb_analysis_combined(
                data_dir, 
                machine, 
                date, 
                test
            )
            df[test + '_legacy'] = [val, unc]
            try:
                if 'TQ' in test:
                    dim = 4
                elif 'SQ' in test:
                    dim = 2
                val_leakage, unc_leakage = rb_analysis_combined(
                    data_dir, 
                    machine, 
                    date, 
                    test, 
                    'leakage_postselect'
                )
                df[test+'_leakage'] = [val_leakage, unc_leakage]
                df[test] = [
                    val + val_leakage/dim, 
                    np.sqrt(unc**2 + unc_leakage**2/dim**2)
                ]

            except KeyError:
                pass
        elif test =='Measurement_crosstalk' or test == 'Reset_crosstalk':
            val, unc = decay_analysis_combined(
                data_dir, 
                machine, 
                date, 
                test
            )
            df[test] = [val, unc]
        elif test == 'SPAM':
            val, unc, res, res_unc = spam_combined(
                data_dir, 
                machine, 
                date, 
                test
            )
            df[test+'0'] = [1-res['0'], res_unc[0]]
            df[test+'1'] = [1-res['1'], res_unc[1]]
            df[test] = [val, unc]
        
    return df


def first_sig_fig(val, unc):

    est = '{:.1uE}'.format(ufloat(val, unc))

    vals, pow = est.split('E')

    sig_val = vals.split('+')[0][1:]

    return float(sig_val + 'E' + pow)
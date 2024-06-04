# Copyright 2024 Quantinuum (www.quantinuum.com)
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

''' Functions for loading RB data from Quantinuum. '''

import json

def load_data(data_dir, machine, date, data_type):

    file_name = f'{machine}/{date}/{data_type}.json'

    with open(data_dir.joinpath(file_name), 'r') as f:
        data = json.load(f)

    return data

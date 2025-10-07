# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from models import ncsnpp, ddpm, ncsnv2, simple_mlp, quasipotential_mlp

def create_model(config):
    """Create the model."""
    model_name = config.model.name
    if model_name == 'ncsnpp':
        return ncsnpp.NCSNpp(config)
    elif model_name == 'ddpm':
        return ddpm.DDPM(config)
    elif model_name == 'ncsnv2':
        return ncsnv2.NCSNv2(config)
    elif model_name == 'simple_mlp':
        return simple_mlp.create_model(config)
    elif model_name == 'quasipotential_mlp':
        return quasipotential_mlp.create_model(config)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
# This file was written by the SMAC3 dev team: https://github.com/automl/smac3

from typing import Optional, Union

import json

from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as csjson
from omegaconf import DictConfig, ListConfig


class JSONCfgEncoder(json.JSONEncoder):
    """Encode DictConfigs.

    Convert DictConfigs to normal dicts.
    """

    def default(self, obj):
        if isinstance(obj, DictConfig):
            return dict(obj)
        elif isinstance(obj, ListConfig):
            parsed_list = []
            for o in obj:
                if type(o) == DictConfig:
                    o = dict(o)
                elif type(o) == ListConfig:
                    o = list(o)
                parsed_list.append(o)

            return parsed_list  # [dict(o) for o in obj]
        return json.JSONEncoder.default(self, obj)


def search_space_to_config_space(
    search_space: Union[str, DictConfig, ConfigurationSpace], seed: Optional[int] = None
) -> ConfigurationSpace:
    """
    Convert hydra search space to SMAC's configuration space.

    See the [ConfigSpace docs](https://automl.github.io/ConfigSpace/master/API-Doc.html#) for information of how
    to define a configuration (search) space.

    In a yaml (hydra) config file, the smac.search space must take the form of:

    search_space:
        hyperparameters:
            hyperparameter_name_0:
                key1: value1
                ...
            hyperparameter_name_1:
                key1: value1
                key2: value2
                ...


    Parameters
    ----------
    search_space : Union[str, DictConfig, ConfigurationSpace]
        The search space, either a DictConfig from a hydra yaml config file, or a path to a json configuration space
        file in the format required of ConfigSpace.
        If it already is a ConfigurationSpace, just optionally seed it.
    seed : Optional[int]
        Optional seed to seed configuration space.


    Example of a json-serialized ConfigurationSpace file.
    {
      "hyperparameters": [
        {
          "name": "x0",
          "type": "uniform_float",
          "log": false,
          "lower": -512.0,
          "upper": 512.0,
          "default": -3.0
        },
        {
          "name": "x1",
          "type": "uniform_float",
          "log": false,
          "lower": -512.0,
          "upper": 512.0,
          "default": -4.0
        }
      ],
      "conditions": [],
      "forbiddens": [],
      "python_module_version": "0.4.17",
      "json_format_version": 0.2
    }


    Returns
    -------
    ConfigurationSpace
    """
    if type(search_space) == str:
        with open(search_space, "r") as f:
            jason_string = f.read()
        cs = csjson.read(jason_string)
    elif type(search_space) == DictConfig:
        # reorder hyperparameters as List[Dict]
        hyperparameters = []
        for name, cfg in search_space.hyperparameters.items():
            cfg["name"] = name
            if "default" not in cfg:
                cfg["default"] = None
            if "log" not in cfg:
                cfg["log"] = False
            if "q" not in cfg:
                cfg["q"] = None
            hyperparameters.append(cfg)
        search_space.hyperparameters = hyperparameters

        if "conditions" not in search_space:
            search_space["conditions"] = []

        if "forbiddens" not in search_space:
            search_space["forbiddens"] = []

        jason_string = json.dumps(search_space, cls=JSONCfgEncoder)
        import logging

        logging.info(f"json string: {jason_string}")
        cs = csjson.read(jason_string)
    elif type(search_space) == ConfigurationSpace:
        cs = search_space
    else:
        raise ValueError(f"search_space must be of type str or DictConfig. Got {type(search_space)}.")

    if seed is not None:
        cs.seed(seed=seed)
    return cs

from importlib import import_module
from typing import Callable, Dict, Any
import inspect

_SIMULATORS_CACHE: Dict[str, Callable[[Dict[str, Any]], "pd.DataFrame"]] = {}

def get_simulator(name: str) -> Callable[[Dict[str, Any]], "pd.DataFrame"]:
    """
    Load a simulator by module name (file name without .py) and
    return its `simulate(params: dict) -> pandas.DataFrame` function.
    """
    if name in _SIMULATORS_CACHE:
        return _SIMULATORS_CACHE[name]

    mod = import_module(f"dgps.{name}")
    if not hasattr(mod, "simulate") or not callable(mod.simulate):
        raise AttributeError(f"Module dgps.{name} must expose a callable `simulate(params: dict) -> DataFrame`.")

    
    sig = inspect.signature(mod.simulate)
    if list(sig.parameters.keys()) != ["params"]:
        raise TypeError(f"`simulate` in dgps.{name} must take exactly one argument named `params`.")

    _SIMULATORS_CACHE[name] = mod.simulate
    return _SIMULATORS_CACHE[name]

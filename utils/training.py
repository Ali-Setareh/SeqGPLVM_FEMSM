import importlib, hashlib, json, torch
from pathlib import Path
from gpytorch.likelihoods import Likelihood 

def class_to_id(cls) -> str:
    """
    Return a fully qualified import path for a class.
    Example: package.module.ClassName
    Used to serialize class references to JSON.
    """
    return f"{cls.__module__}.{cls.__name__}" 

def id_to_class(s: str):
    """
    Inverse of class_to_id.
    Dynamically imports the module and returns the class object.
    """
    mod, _, name = s.rpartition(".")
    return getattr(importlib.import_module(mod), name)

def dump_train_cfg_json(path: Path, cfg: dict) -> None:
    """
    Serialize a training configuration dictionary to JSON.

    Special handling:
      - If 'treatment_model' is a class object, it is replaced by its import path string.
      - If 'init_z' is a torch.Tensor, it is saved separately to 'init_z.pt' in the same
        directory as 'path' and replaced by a JSON object: {"path": "<file path>"}.

    Parameters:
      path: Full file path (including filename) where JSON will be written.
      cfg:  Configuration dictionary (will not be mutated).
    """
    cfg = cfg.copy()
    # class -> import path
    tm = cfg.get("treatment_model")
    if isinstance(tm, type):
        cfg["treatment_model"] = class_to_id(tm)

    # tensor -> file reference
    iz = cfg.get("init_z")
    if isinstance(iz, torch.Tensor):
        iz_path = path / f"init_z.pt"
        torch.save(iz.detach().cpu(), iz_path)
        cfg["init_z"] = {"path": str(iz_path)}
    path.write_text(json.dumps(cfg))

def load_train_cfg_from_json(path: Path, device: str | torch.device = "cpu") -> dict:
    """
    Load and reconstruct a training configuration previously saved with dump_train_cfg_json.

    Restores:
      - 'treatment_model' import path string back to the class object.
      - 'init_z' path reference back to a torch.Tensor on the specified device.

    Parameters:
      path:   Path to the JSON config file.
      device: Device mapping for any restored tensors.

    Returns:
      A configuration dictionary with objects restored.
    """
    cfg = json.loads(Path(path).read_text())
    # import path -> class
    tm = cfg.get("treatment_model")
    if isinstance(tm, str):
        cfg["treatment_model"] = id_to_class(tm)
    # file reference -> tensor
    iz = cfg.get("init_z")
    if isinstance(iz, dict) and "path" in iz:
        cfg["init_z"] = torch.load(iz["path"], map_location=device)
    return cfg

def resolve_treatment_model(spec):
    if spec is None:
        return None
    if isinstance(spec, type) and issubclass(spec, Likelihood):
        return spec
    
    raise ValueError(f"Unsupported treatment_model spec: {spec!r}")

def materialize_cfg(cfg: dict, device: torch.device) -> dict:
    cfg = dict(cfg)
    # class / alias / import path -> class
    cfg["treatment_model"] = resolve_treatment_model(cfg.get("treatment_model"))
    # {"path": "..."} -> tensor
    iz = cfg.get("init_z")
    if isinstance(iz, dict) and "path" in iz:
        cfg["init_z"] = torch.load(iz["path"], map_location=device)
    return cfg
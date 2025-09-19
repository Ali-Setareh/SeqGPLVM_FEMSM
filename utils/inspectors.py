import torch
def get_actuals_via_getters(module: torch.nn.Module) -> dict:
    """
    Scans `module` (and its submodules) for any Parameter named 'raw_*',
    then uses the matching getter (stripping 'raw_') to fetch the actual value.
    
    Returns
    -------
    actuals : dict
        Mapping from the raw-parameter fullname (e.g.
        'base_kernel.raw_lengthscale') → actual Tensor value.
    """
    actuals = {}
    for raw_name, _ in module.named_parameters():                 # recursion built in
        if not raw_name.startswith("raw_") and ".raw_" not in raw_name:
            continue

        # Split the hierarchy so we can reach the owning submodule:
        parts = raw_name.split(".")
        submod = module
        for comp in parts[:-1]:
            submod = getattr(submod, comp)

        # Derive the getter name: drop the 'raw_' prefix
        raw_attr    = parts[-1]                   # e.g. 'raw_lengthscale'
        actual_attr = raw_attr.replace("raw_", "")  # -> 'lengthscale'

        # If that property exists on the submodule, grab it:
        if hasattr(submod, actual_attr):
            actuals[raw_name] = getattr(submod, actual_attr).detach().cpu().numpy()
        # otherwise we silently skip (you could log a warning here)
    return actuals
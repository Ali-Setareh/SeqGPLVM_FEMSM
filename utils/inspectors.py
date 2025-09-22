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

import plotly.graph_objects as go
import numpy as np

def plot_param_history(param_hist: dict,
                       key: str,
                       ls_num: int = 0,
                       width: int = 1000,
                       height: int = 500, 
                       x_start: int | float = 0,      # ← add
                       x_step: int | float = 1) -> go.Figure:
    """
    Plot the history of parameters stored in param_hist.

    Parameters
    ----------
    param_hist : dict
        A dict mapping strings (like 'layer1.var', 'layer2.var', etc.) to
        history arrays of shape (iterations, ...) or higher.
    key : str
        The suffix of the parameter names you want to plot (e.g. 'var', 'ls',
        'z_mu', 'z_logsigma', 'intercept', 'noise_var').
    ls_num : int, optional
        Only used if key == 'ls': which index along the last axis to plot.
    width, height : int, optional
        Figure dimensions.

    Returns
    -------
    fig : go.Figure
        The Plotly figure object (also shown inline).
    """
    # 1) collect matching arrays
    
    
    if key in ('Z_val.q_mu', 'Z_val.q_log_sigma','Z.q_mu', 'Z.q_log_sigma'):
        arrays = param_hist[key]
    else: 
        arrays = [v for k,v in param_hist.items() if k.split('.')[-1] == key]
    if not arrays:
        raise KeyError(f"No entries in param_hist end with '.{key}'")

    plot_list = np.array(arrays).squeeze()

    n_iters = plot_list.shape[1] if plot_list.ndim >= 2 else plot_list.shape[0]
    x_vals = x_start + np.arange(n_iters) * x_step

    # 2) for z_mu / z_logsigma we want to transpose so “i‐over‐iterations” matches
    if key in ('Z_val.q_mu', 'Z_val.q_log_sigma','Z.q_mu', 'Z.q_log_sigma'):
        
        plot_list = plot_list.T

    # 3) build the figure
    fig = go.FigureWidget()  #go.Figure()
    if key in ("ls","raw_lengthscale"):
        # plot_list shape: (n_layers, n_iters, n_ls)
        for i in range(plot_list.shape[0]):
            y = plot_list[i, :, ls_num]
            fig.add_trace(go.Scatter(
                y=y,
                x=x_vals,
                mode='lines',            
                #marker=dict(size=8, opacity=0),
                name=f"{key}_{i}"
            ))
        title_suffix = f"_{ls_num}"
    else:
        # plot_list shape: (n_layers, n_iters)
        for i in range(plot_list.shape[0]):
            y = plot_list[i, :]
            fig.add_trace(go.Scatter(
                y=y,
                x=x_vals,
                mode='lines',
                name=f"{key}_{i}"
            ))
        title_suffix = ""

    # 4) one layout call
    fig.update_layout(
        title=dict(
            text=f"{key}{title_suffix} over Iterations",
            x=0.5, font=dict(size=20)
        ),
        xaxis=dict(
            title=dict(text="Iteration", font=dict(size=16)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text=(key + title_suffix).capitalize(),
                font=dict(size=16)
            ),
            tickfont=dict(size=12)
        ),
        legend=dict(
            y=1, x=1.02,
            traceorder="normal",
            font=dict(size=10),
            bordercolor="Black",
            borderwidth=1,
            itemsizing='constant',
            valign='middle',
            title=dict(
                text=(key + title_suffix).capitalize(),
                font=dict(family="Arial")), 
            # ← add these two lines:
            itemclick="toggleothers",     # single‑click a legend item → hide all the others
            #itemdoubleclick="reset"       # double‑click the same item → restore everything

        ),
        width=width,
        height=height,
        margin=dict(r=200, t=80),
        clickmode="event+select"
    )

    fig.show()
    return fig

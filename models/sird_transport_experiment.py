import numpy as np

from pathlib import Path
from sklearn.model_selection import ParameterGrid

from utils import grid_experiment
from sird_transport import run, sird_transport_model

model_name = "SIRD_T_time1"
N1 = 1e7
N2 = 5e6
parameters = {
    "beta1": 0.45,
    "omega1": 0.05,
    "gamma1": 0.0294,
    "beta2": 0.4,
    "omega2": 0.05,
    "gamma2": 0.0294,
    "tau12": 0.02,
    "tau21": 0.01,
}

hyperparam_grid = ParameterGrid(
    {
        "time_range": [(0, 366)],
        "time_step": [1],
        "search_range":[(0.2, 1.8)],
        "iterations":[30000],
        "layers":[3, 5],
        "neurons":[32, 64, 128],
        "activation":["relu"],
        "loss_weights":[
            8 * [1] + 8 * [1] + 8 * [1],
            8 * [1e1] + 8 * [1] + 8 * [1],
            8 * [1] + 8 * [1] + 8 * [1e1],
            8 * [1e1] + 8 * [1] + 8 * [1e1]
        ],
    }
)
output_path = Path() / "output" / model_name
output_path.mkdir(parents=True, exist_ok=True)
t_pred = (
    np.arange(
        start=hyperparam_grid[0]["time_range"][0],
        stop=hyperparam_grid[0]["time_range"][1],
        step=1
    )[:, np.newaxis]
)
error_grid = grid_experiment(
    run=run,
    t_pred=t_pred,
    ode_solver=sird_transport_model,
    N=(N1, N2),
    parameters=parameters,
    hyperparam_grid=hyperparam_grid,
    output_path=output_path
)
print(error_grid)
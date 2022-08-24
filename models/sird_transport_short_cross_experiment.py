import numpy as np

from pathlib import Path
from sklearn.model_selection import ParameterGrid

from utils import grid_experiment
from sird_transport_short_cross import run, sird_transport_model


model_name = "SIRD_TS"
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
    "delta12": 0.01,
    "delta21": 0.01,
    "zeta12": 0.01,
    "zeta21": 0.01,
}
t_train = np.arange(0, 366, 3)[:, np.newaxis]
t_pred =  np.arange(0, 366, 1)[:, np.newaxis]

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
error_grid = grid_experiment(
    run=run,
    ode_solver=sird_transport_model,
    t_train=t_train,
    t_pred=t_pred,
    N=(N1, N2),
    parameters=parameters,
    hyperparam_grid=hyperparam_grid,
    output_path=output_path
)
print(error_grid)









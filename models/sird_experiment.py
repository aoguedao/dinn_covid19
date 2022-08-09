import numpy as np

from pathlib import Path
from sklearn.model_selection import ParameterGrid

from utils import grid_experiment
from sird import run


model_name = "SIRD"
N = 1e6
parameters = {
    "beta": 0.5,
    "omega": 1 / 14,
    "gamma": 1 / 5,
}
t_train = np.arange(0, 366, 3)[:, np.newaxis]
t_pred =  np.arange(0, 366, 1)[:, np.newaxis]

hyperparam_grid = ParameterGrid(
    {
        "search_range":[(0.2, 1.8)],
        "iterations":[30000, 50000],
        "layers":[3, 5],
        "neurons":[32, 64, 128],
        "activation":["relu"],
        "loss_weights":[
            4 * [1] + 4 * [1] + 4 * [1],
            4 * [1e1] + 4 * [1] + 4 * [1],
            4 * [1] + 4 * [1] + 4 * [1e1],
            4 * [1e1] + 4 * [1] + 4 * [1e1],
        ],
    }
)
output_path = Path() / "output" / model_name
output_path.mkdir(parents=True, exist_ok=True)
error_grid = grid_experiment(
    run=run,
    t_train=t_train,
    t_pred=t_pred,
    N=N,
    parameters=parameters,
    hyperparam_grid=hyperparam_grid,
    output_path=output_path
)
print(error_grid)

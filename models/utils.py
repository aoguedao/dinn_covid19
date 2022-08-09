import datetime

import numpy as np
import pandas as pd


def grid_experiment(
    run,
    t_train,
    t_pred,
    N,
    parameters,
    hyperparam_grid,
    output_path
    ):

    hyperparam_df = (
        pd.DataFrame(
            hyperparam_grid,
            index=np.arange(len(hyperparam_grid)) + 1
        )
        .rename_axis("run")
        .reset_index()
        .assign(
            # loss_weights=lambda df: df["loss_weights"].apply(lambda x: ",".join(str(y) for y in x))
            loss_weights=lambda df: df["loss_weights"].apply(tuple)
        )
    )

    error_dict = {}
    for i, hyperparameters in enumerate(hyperparam_grid, 1):
        print(f"Iteration {i} at {datetime.datetime.now()}")
        print(hyperparameters)
        if isinstance(N, tuple) and len(N) ==2:
            N1, N2 = N
            error_df, _ = run(
                t_train=t_train,
                t_pred=t_pred,
                N1=N1,
                N2=N2,
                parameters=parameters,
                hyperparameters=hyperparameters,
                filepath=output_path / f"{output_path.name}_estimation_run_{i:02d}.png"
            )
        else:
            error_df, _ = run(
                t_train=t_train,
                t_pred=t_pred,
                N=N,
                parameters=parameters,
                hyperparameters=hyperparameters,
                filepath=output_path / f"{output_path.name}_estimation_run_{i:02d}.png"
            )
        error_dict[i] = error_df

        error_grid = (
            pd.concat(error_dict)
            .rename_axis(["run", "parameter"])
            .reset_index()
            .merge(hyperparam_df, how="right", on="run")
            .set_index(
                [
                    "run",
                    "search_range",
                    "iterations",
                    "layers",
                    "neurons",
                    "activation",
                    "loss_weights",
                    "parameter"
                ]
            )
        )
        error_grid.to_excel(output_path / f"{output_path.name}_grid_parameter_estimation.xlsx")

    return error_grid
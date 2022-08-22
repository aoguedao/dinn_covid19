import datetime
import numpy as np
import pandas as pd
import deepxde as dde


def grid_experiment(
    run,
    ode_solver,
    t_train,
    t_pred,
    N,
    parameters,
    hyperparam_grid,
    output_path
    ):

    idx_list = [
        "run",
        "search_range",
        "iterations",
        "layers",
        "neurons",
        "activation",
        "loss_weights"
    ]

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

    metric_dict = {}
    error_dict = {}
    for i, hyperparameters in enumerate(hyperparam_grid, 1):
        print(f"Iteration {i} at {datetime.datetime.now()}")
        print(hyperparameters)
        if isinstance(N, tuple) and len(N) == 2:
            N1, N2 = N
            y_true = ode_solver(np.ravel(t_pred), N1, N2, parameters)
            model, error_df, _ = run(
                t_train=t_train,
                t_pred=t_pred,
                N1=N1,
                N2=N2,
                parameters=parameters,
                hyperparameters=hyperparameters,
                filepath=output_path / f"{output_path.name}_estimation_run_{i:02d}.png"
            )
        else:
            y_true = ode_solver(np.ravel(t_pred), N, parameters)
            error_df, _ = run(
                t_train=t_train,
                t_pred=t_pred,
                N=N,
                parameters=parameters,
                hyperparameters=hyperparameters,
                filepath=output_path / f"{output_path.name}_estimation_run_{i:02d}.png"
            )
        y_pred = model.predict(t_pred)

        metric_dict[i] = pd.DataFrame(
            {
                "mean_squared_error": [dde.metrics.mean_squared_error(y_true, y_pred)],
                "mean_l2_relative_error": [dde.metrics.mean_l2_relative_error(y_true, y_pred)],
                "mean_absolute_percentage_error": [dde.metrics.mean_absolute_percentage_error(y_true, y_pred)],
                "l2_relative_error": [dde.metrics.l2_relative_error(y_true, y_pred)]

            }
        )
        error_dict[i] = error_df

        metric_grid = (
            pd.concat(metric_dict)
            .droplevel(1)
            .rename_axis("run")
            .reset_index()
            .merge(hyperparam_df, how="right", on="run")
            .set_index(idx_list)
        )
        error_grid = (
            pd.concat(error_dict)
            .rename_axis(["run", "parameter"])
            .reset_index()
            .merge(hyperparam_df, how="right", on="run")
            .set_index(idx_list + ["parameter"])
        )

        with pd.ExcelWriter(output_path / f"{output_path.name}_grid_estimation.xlsx") as writer:  
            metric_grid.to_excel(writer, sheet_name="Metrics")
            error_grid.to_excel(writer, sheet_name="Parameters")
    return error_grid
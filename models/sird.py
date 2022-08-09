import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import deepxde as dde
import seaborn as sns

from deepxde.backend import tf
from scipy.integrate import odeint
from math import tanh

sns.set_theme(style="darkgrid")

# dde.config.real.set_float64()


def sird_model(
    t,
    N,
    parameters
):
    beta = parameters["beta"]
    omega = parameters["omega"]
    gamma = parameters["gamma"]

    def func(y, t):
        S, I, R, D = y
        dS = - beta / N * S * I
        dI = beta / N * S * I - gamma * I - omega * I
        dR = gamma * I
        dD =  omega * I
        return [dS, dI, dR, dD]
    
    S_0 = N - 1
    I_0 = 1
    R_0 = 0
    D_0 = 0
    y0 = [S_0, I_0, R_0, D_0]
    return odeint(func, y0, t)
    

def dinn(
    data_t,
    data_y,
    N,
    parameters,
    hyperparameters={
        "search_range": (0.2, 1.8),
        "iterations": 30000,
        "layers": 3,
        "neurons": 64,
        "activation": "relu",
        "loss_weights": None
    }
    
):    
        
    def get_variable_in_search_range(nominal , var, search_range):
        low = nominal * search_range[0]
        up = nominal * search_range[1]
        scale = (up - low) / 2
        tanh_var = tf.tanh(var) if isinstance(var, tf.Variable) else tanh(var)
        return scale * tanh_var + scale + low

    # Variables
    _beta = dde.Variable(0.0)
    _omega = dde.Variable(0.0)
    _gamma = dde.Variable(0.0)
    variables = [
        _beta,
        _omega,
        _gamma,
    ]
    
    # ODE model
    def ODE(t, y):
        S = y[:, 0:1]
        I = y[:, 1:2]
        R = y[:, 2:3]
        D = y[:, 3:4]
    
        beta = get_variable_in_search_range(parameters["beta"], _beta, hyperparameters["search_range"])
        omega = get_variable_in_search_range(parameters["omega"], _omega, hyperparameters["search_range"])
        gamma = get_variable_in_search_range(parameters["gamma"], _gamma, hyperparameters["search_range"])

        dS_t = dde.grad.jacobian(y, t, i=0)
        dI_t = dde.grad.jacobian(y, t, i=1)
        dR_t = dde.grad.jacobian(y, t, i=2)
        dD_t = dde.grad.jacobian(y, t, i=3)

        return [
            dS_t - (- beta / N * S * I),
            dI_t - (beta / N * S * I - gamma * I - omega * I),
            dR_t - (gamma * I),
            dD_t - (omega * I)
        ]
    
    # Geometry
    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])
    
    # Boundaries
    def boundary(_, on_initial):
        return on_initial
    
    # Initial conditions
    ic_S = dde.icbc.IC(geom, lambda x: N- 1, boundary, component=0)
    ic_I = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
    ic_R = dde.icbc.IC(geom, lambda x: 0, boundary, component=2)
    ic_D = dde.icbc.IC(geom, lambda x: 0, boundary, component=3)

    # Train data
    observe_S = dde.icbc.PointSetBC(data_t, data_y[:, 0:1], component=0)
    observe_I = dde.icbc.PointSetBC(data_t, data_y[:, 1:2], component=1)
    observe_R = dde.icbc.PointSetBC(data_t, data_y[:, 2:3], component=2)
    observe_D = dde.icbc.PointSetBC(data_t, data_y[:, 3:4], component=3)
    
    # Model
    data = dde.data.PDE(
        geom,
        ODE,
        [
            ic_S,
            ic_I,
            ic_R,
            ic_D,
            observe_S,
            observe_I,
            observe_R,
            observe_D
        ],
        num_domain=0,
        num_boundary=2,
        anchors=data_t,
    )
    
    neurons = hyperparameters["neurons"]
    layers = hyperparameters["layers"]
    activation = hyperparameters["activation"]
    net = dde.nn.FNN([1] + [neurons] * layers + [4], activation, "Glorot uniform")
    
    def feature_transform(t):
        t = t / data_t[-1, 0]
        return t

    net.apply_feature_transform(feature_transform)

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=hyperparameters["loss_weights"],
        external_trainable_variables=variables
    )
    variable = dde.callbacks.VariableValue(
        variables,
        period=5000,
    )
    model.train(
        iterations=hyperparameters["iterations"],
        display_every=5000,
        callbacks=[variable]
    )

    parameters_pred = {
        name: get_variable_in_search_range(nominal, var, hyperparameters["search_range"])
        for (name, nominal), var in zip(parameters.items(), variable.value)
    }

    return model, parameters_pred


def error(parameters, parameters_pred):
    errors = (
        pd.DataFrame(
            {
                "Real": parameters,
                "Predicted": parameters_pred
            }
        )
        .assign(
            **{"Relative Error": lambda x: (x["Real"] - x["Predicted"]).abs() / x["Real"]}
        )
    )
    return errors


def plot(data_pred, data_real, filepath):

    g = sns.relplot(
        data=data_pred,
        x="time",
        y="population",
        hue="status",
        kind="line",
        aspect=2,
    )

    sns.scatterplot(
        data=data_real,
        x="time",
        y="population",
        hue="status",
        ax=g.ax,
        legend=False
    )

    (
        g.set_axis_labels("Time", "Population")
        .tight_layout(w_pad=1)
    )

    g._legend.set_title("Status")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"SIRD model estimation")
    if filepath is not None:
        plt.savefig(filepath, dpi=300)
    else:
        plt.close()
    return g


def run(
    t_train,
    t_pred,
    N,
    parameters,
    hyperparameters,
    filepath=None
):

    populations_names = ["S", "I", "R", "D"]

    y_train = sird_model(np.ravel(t_train), N, parameters)
    data_real = (
        pd.DataFrame(y_train, columns=populations_names, index=t_train.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status", value_name="population")
    )

    model, parameters_pred = dinn(
        data_t=t_train,
        data_y=y_train,
        N=N,
        parameters=parameters,
        hyperparameters=hyperparameters
    )
    
    y_pred = model.predict(t_pred)
    data_pred = (
        pd.DataFrame(y_pred, columns=populations_names, index=t_pred.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status", value_name="population")
    )

    error_df = error(parameters, parameters_pred)
    fig = plot(data_pred, data_real, filepath)

    return error_df, fig


if __name__ == "__main__":
    N = 1e6
    parameters = {
        "beta": 0.5,
        "omega": 1 / 14,
        "gamma": 1 / 5,
    }
    hyperparameters = {
        "search_range": (0.2, 1.8),
        "iterations": 30000,
        "layers": 3,
        "neurons": 64,
        "activation": "relu",
        "loss_weights": 4 * [1] + 4 * [1] + 4 * [1],
    }
    t_train = np.arange(0, 366, 3)[:, np.newaxis]
    t_pred =  np.arange(0, 366, 1)[:, np.newaxis]
    error_df, fig = run(
        t_train=t_train,
        t_pred=t_pred,
        N=N,
        parameters=parameters,
        hyperparameters=hyperparameters,
        filepath=None
    )
    print(error_df)
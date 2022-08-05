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


def sird_transport_model(
    t,
    N1,
    N2,
    parameters
):
    beta1 = parameters["beta1"]
    omega1 = parameters["omega1"]
    gamma1 = parameters["gamma1"]
    beta2 = parameters["beta2"]
    omega2 = parameters["omega2"]
    gamma2 = parameters["gamma2"]
    tau12 = parameters["tau12"]
    tau21 = parameters["tau21"]

    def func(y, t):
        S1, I1, R1, D1, S2, I2, R2, D2 = y
        return [
            - beta1 / N1 * S1 * I1 - tau12 * S1 + tau21 * S2,
            beta1 / N1 * S1 * I1 - omega1 * I1 - gamma1 * I1 - tau12 * I1 + tau21 * I2,
            omega1 * I1 - tau12 * R1 + tau21 * R2,
            gamma1 * I1,
            - beta2 / N2 * S2 * I2 - tau21 * S2 + tau12 * S1,
            beta2 / N2 * S2 * I2 - omega2 * I2 - gamma2 * I2  - tau21 * I2 + tau12 * I1,
            omega2 * I2 - tau21 * R2 + tau12 * R1,
            gamma2 * I2
        ]
    
    S1_0 = N1 - 1
    I1_0 = 1
    R1_0 = 0
    D1_0 = 0
    S2_0 = N2 - 1
    I2_0 = 1
    R2_0 = 0
    D2_0 = 0
    y0 = [S1_0, I1_0, R1_0, D1_0, S2_0, I2_0, R2_0, D2_0]
    return odeint(func, y0, t)
    

def dinn(
    data_t,
    data_y,
    N1,
    N2,
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
    _beta1 = dde.Variable(0.0)
    _omega1 = dde.Variable(0.0)
    _gamma1 = dde.Variable(0.0)
    _beta2 = dde.Variable(0.0)
    _omega2 = dde.Variable(0.0)
    _gamma2 = dde.Variable(0.0)
    _tau12 = dde.Variable(0.0)
    _tau21 = dde.Variable(0.0)
    variables = [
        _beta1,
        _omega1,
        _gamma1,
        _beta2,
        _omega2,
        _gamma2,
        _tau12,
        _tau21
    ]
    
    # ODE model
    def ODE(t, y):
        S1 = y[:, 0:1]
        I1 = y[:, 1:2]
        R1 = y[:, 2:3]
        D1 = y[:, 3:4]
        S2 = y[:, 4:5]
        I2 = y[:, 5:6]
        R2 = y[:, 6:7]
        D2 = y[:, 7:8]
    
        beta1 = get_variable_in_search_range(parameters["beta1"], _beta1, hyperparameters["search_range"])
        omega1 = get_variable_in_search_range(parameters["omega1"], _omega1, hyperparameters["search_range"])
        gamma1 = get_variable_in_search_range(parameters["gamma1"], _gamma1, hyperparameters["search_range"])
        beta2 = get_variable_in_search_range(parameters["beta2"], _beta2, hyperparameters["search_range"])
        omega2 = get_variable_in_search_range(parameters["omega2"], _omega2, hyperparameters["search_range"])
        gamma2 = get_variable_in_search_range(parameters["gamma2"], _gamma2, hyperparameters["search_range"])
        tau12 = get_variable_in_search_range(parameters["tau12"], _tau12, hyperparameters["search_range"])
        tau21 = get_variable_in_search_range(parameters["tau21"], _tau21, hyperparameters["search_range"])

        dS1_t = dde.grad.jacobian(y, t, i=0)
        dI1_t = dde.grad.jacobian(y, t, i=1)
        dR1_t = dde.grad.jacobian(y, t, i=2)
        dD1_t = dde.grad.jacobian(y, t, i=3)
        dS2_t = dde.grad.jacobian(y, t, i=4)
        dI2_t = dde.grad.jacobian(y, t, i=5)
        dR2_t = dde.grad.jacobian(y, t, i=6)
        dD2_t = dde.grad.jacobian(y, t, i=7)
        
        return [
            dS1_t - (- beta1 / N1 * S1 * I1 - tau12 * S1 + tau21 * S2),
            dI1_t - (beta1 / N1 * S1 * I1 - omega1 * I1 - gamma1 * I1 - tau12 * I1 + tau21 * I2),
            dR1_t - (omega1 * I1 - tau12 * R1 + tau21 * R2),
            dD1_t - (gamma1 * I1),
            dS2_t - (- beta2 / N2 * S2 * I2 - tau21 * S2 + tau12 * S1),
            dI2_t - (beta2 / N2 * S2 * I2 - omega2 * I2 - gamma2 * I2  - tau21 * I2 + tau12 * I1),
            dR2_t - (omega2 * I2 - tau21 * R2 + tau12 * R1),
            dD2_t - (gamma2 * I2)
        ]
    
    # Geometry
    geom = dde.geometry.TimeDomain(data_t[0, 0], data_t[-1, 0])
    
    # Boundaries
    def boundary(_, on_initial):
        return on_initial
    
    # Initial conditions
    ic_S1 = dde.icbc.IC(geom, lambda x: N1- 1, boundary, component=0)
    ic_I1 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
    ic_R1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=2)
    ic_D1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=3)
    ic_S2 = dde.icbc.IC(geom, lambda x: N2- 1, boundary, component=4)
    ic_I2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=5)
    ic_R2 = dde.icbc.IC(geom, lambda x: 0, boundary, component=6)
    ic_D2 = dde.icbc.IC(geom, lambda x: 0, boundary, component=7)

    # Train data
    observe_S1 = dde.icbc.PointSetBC(data_t, data_y[:, 0:1], component=0)
    observe_I1 = dde.icbc.PointSetBC(data_t, data_y[:, 1:2], component=1)
    observe_R1 = dde.icbc.PointSetBC(data_t, data_y[:, 2:3], component=2)
    observe_D1 = dde.icbc.PointSetBC(data_t, data_y[:, 3:4], component=3)
    observe_S2 = dde.icbc.PointSetBC(data_t, data_y[:, 4:5], component=4)
    observe_I2 = dde.icbc.PointSetBC(data_t, data_y[:, 5:6], component=5)
    observe_R2 = dde.icbc.PointSetBC(data_t, data_y[:, 6:7], component=6)
    observe_D2 = dde.icbc.PointSetBC(data_t, data_y[:, 7:8], component=7)
    
    # Model
    data = dde.data.PDE(
        geom,
        ODE,
        [
            ic_S1,
            ic_I1,
            ic_R1,
            ic_D1,
            ic_S2,
            ic_I2,
            ic_R2,
            ic_D2,
            observe_S1,
            observe_I1,
            observe_R1,
            observe_D1,
            observe_S2,
            observe_I2,
            observe_R2,
            observe_D2
        ],
        num_domain=0,
        num_boundary=2,
        anchors=data_t,
    )
    
    neurons = hyperparameters["neurons"]
    layers = hyperparameters["layers"]
    activation = hyperparameters["activation"]
    net = dde.nn.FNN([1] + [neurons] * layers + [8], activation, "Glorot uniform")
    
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


def plot(data_pred, data_real):
    
    sns.set(rc={"figure.facecolor":"white"})
    
    g = sns.relplot(
        data=data_pred,
        x="time",
        y="population",
        row="region",
        hue="status",
        kind="line",
        aspect=2.5,
    )

    for region, ax in g.axes_dict.items():
        sns.scatterplot(
            data=data_real.query("region == @region"),
            x="time",
            y="population",
            hue="status",
            ax=ax,
            legend=False
        )


    (
        g.set_axis_labels("Time", "Population")
        .set_titles("Region {row_name}")
        .tight_layout(w_pad=1)
    )

    g._legend.set_title("Status")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Estimation SIRD model with transportation between two regions")
    plt.savefig("SIRD_transport_estimation.png", dpi=300)
    return g


def run(
    t_train,
    t_pred,
    N1,
    N2,
    parameters,
    hyperparameters
):

    populations_names = ["S1", "I1", "R1", "D1", "S2", "I2", "R2", "D2"]


    y_train = sird_transport_model(np.ravel(t_train), N1, N2, parameters)
    data_real = (
        pd.DataFrame(y_train, columns=populations_names, index=t_train.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status_region", value_name="population")
        .assign(
            status=lambda x: x["status_region"].str[0],
            region=lambda x: x["status_region"].str[1],
        )
    )

    model, parameters_pred = dinn(
        data_t=t_train,
        data_y=y_train,
        N1=N1,
        N2=N2,
        parameters=parameters,
        hyperparameters=hyperparameters
    )
    
    y_pred = model.predict(t_pred)
    data_pred = (
        pd.DataFrame(y_pred, columns=populations_names, index=t_pred.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status_region", value_name="population")
        .assign(
            status=lambda x: x["status_region"].str[0],
            region=lambda x: x["status_region"].str[1],
        )
    )

    error_df = error(parameters, parameters_pred)
    fig = plot(data_pred, data_real)

    return error_df, fig


if __name__ == "__main__":
    N1 = 1e7
    N2 = 7e6
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
    hyperparameters = {
        "search_range": (0.2, 1.8),
        "iterations": 50000,
        "layers": 3,
        "neurons": 128,
        "activation": "relu",
        "loss_weights": 8 * [1e1] + 8 * [1] + 8 * [1],
    }
    t_train = np.arange(0, 366, 3)[:, np.newaxis]
    t_pred =  np.arange(0, 366, 1)[:, np.newaxis]
    error_df, fig = run(
        t_train=t_train,
        t_pred=t_pred,
        N1=N1,
        N2=N2,
        parameters=parameters,
        hyperparameters=hyperparameters
    )
    print(error_df)
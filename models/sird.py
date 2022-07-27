import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import deepxde as dde
import seaborn as sns

from deepxde.backend import tf
from scipy.integrate import odeint

sns.set_theme(style="darkgrid")

# dde.config.real.set_float64()


def sird_model(
    t,
    N,
    beta,
    omega,
    gamma,
):
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


def dinn(data_t, data_y, N):    
    
    # Variables
    beta = tf.math.sigmoid(dde.Variable(0.1))
    omega = tf.math.sigmoid(dde.Variable(0.1))
    gamma = tf.math.sigmoid(dde.Variable(0.1))
    variable_list = [beta, omega, gamma]
    
    # ODE model
    def ODE(t, y):
        S = y[:, 0:1]
        I = y[:, 1:2]
        R = y[:, 2:3]
        D = y[:, 3:4]
        
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
        num_domain=400,
        num_boundary=2,
        anchors=data_t,
    )
    
    net = dde.nn.FNN([1] + [128] * 3 + [4], "relu", "Glorot uniform")
    
    def feature_transform(t):
        t = t / data_t[-1, 0]
        return t

    # def output_transform(t, y):
    #     # return tf.constant([1, 1, 1e2, 1, 1, 1]) * y
    #     return tf.abs(y)

    net.apply_feature_transform(feature_transform)
    # net.apply_output_transform(output_transform)

    iterations = 50000
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=4 * [1] + 4 * [1] + 4 * [1e1],
        external_trainable_variables=variable_list
    )
    variable = dde.callbacks.VariableValue(
        variable_list,
        period=5000,
        filename="variables_sird.dat"
    )
    losshistory, train_state = model.train(
        iterations=iterations,
        display_every=10000,
        callbacks=[variable]
      )
    # dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    return model, variable


def error(parameters_real, parameters_pred):
    parameter_names = [
        "beta",
        "omega",
        "gamma",
    ]
    errors = (
        pd.DataFrame(
            {
                "real": parameters_real,
                "predicted": parameters_pred
            },
            index=parameter_names
        )
        .assign(
            relative_error=lambda x: (x["real"] - x["predicted"]).abs() / x["real"]
        )
    )
    return errors


def plot(data_pred, data_real):

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
        .set_titles("Zone {row_name}")
        .tight_layout(w_pad=1)
    )

    g._legend.set_title("Status")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"SIRD model estimation")
    return g


def run(N, beta, omega, gamma):

    names = list("SIRD")
    t = np.arange(0, 366, 7)[:, np.newaxis]
    y = sird_model(np.ravel(t), N, beta, omega, gamma)
    data_real = (
        pd.DataFrame(y, columns=names, index=t.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status", value_name="population")
    )

    model, variable = dinn(t, y, N)
    
    full_t = np.arange(0, 366)[:, np.newaxis]
    y_pred = model.predict(full_t)
    data_pred = (
        pd.DataFrame(y_pred, columns=names, index=full_t.ravel())
        .rename_axis("time")
        .reset_index()
        .melt(id_vars="time", var_name="status", value_name="population")
    )

    parameters_real = [beta, omega, gamma]
    parameters_pred = variable.value
    error_df = error(parameters_real, parameters_pred)
    fig = plot(data_pred, data_real)

    return error_df, fig


if __name__ == "__main__":
    N = 1000
    beta = 0.5
    omega = 1 / 14
    gamma = 1 / 5
    error_df, fig = run(N, beta, omega, gamma)
    plt.show()
    print(error_df)
import numpy as np
import streamlit as st

from models.sird import run

st.set_page_config(
     page_title="Disease Informed Neural Networks",
     page_icon="🧬",
     layout="wide",
)

st.title("COVID-19 and Disease Informed Neural Networks")

tab_sird, tab_help = st.tabs(["SIRD", "Help"])

with tab_sird:

    st.latex(r"""
        \begin{align*}
        \frac{dS}{dt} &= - \frac{\beta}{N}  S I \\
        \frac{dI}{dt} &= \frac{\beta}{N} S I - \omega  I - \gamma I \\
        \frac{dR}{dt} &= \omega I \\
        \frac{dD}{dt} &= \gamma I \\
        \end{align*}
    """
    )

    col11, col12, col13, col14 = st.columns(4)

    N = col11.number_input(
        "N",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        format=None,
        help="Population",
    )

    beta = col12.number_input(
        "Beta",
        min_value=0.001,
        max_value=1.0,
        value=0.5,
        step=0.001,
        format=None,
        help="Transmission Rate",
    )

    omega = col13.number_input(
        "Omega",
        min_value=0.001,
        max_value=1.0,
        value=1 / 14,
        step=0.001,
        format=None,
        help="Rate at which Infected individuals become Recovered",
    )

    gamma = col14.number_input(
        "Gamma",
        min_value=0.001,
        max_value=1.0,
        value=1 / 5,
        step=0.001,
        format=None,
        help="Rate at which Infected individuals become Dead",
    )

    
    col21, col22, col23 = st.columns(3)

    iterations = col21.number_input(
        "Iterations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        format=None,
        help="Training iterations",
    )

    layers = col22.number_input(
        "Layers",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        format=None,
        help="Neural Network hidden layers",
    )

    neurons = col23.number_input(
        "Neurons",
        min_value=8,
        max_value=256,
        value=64,
        step=8,
        format=None,
        help="Neurons for each hidden layer",
    )

    if st.button("Run DINN"):
        with st.spinner('Wait for it...'):
            t_train = np.arange(0, 366, 3)[:, np.newaxis]
            t_pred =  np.arange(0, 366, 1)[:, np.newaxis]
            parameters = {
                "beta": beta,
                "omega": omega,
                "gamma": gamma,
            }
            hyperparameters = {
                "search_range": (0.2, 1.8),
                "iterations": iterations,
                "layers": layers,
                "neurons": neurons,
                "activation": "relu",
                "loss_weights": 4 * [1] + 4 * [1] + 4 * [1],
            }
            error_df, fig = run(
                t_train=t_train,
                t_pred=t_pred,
                N=N,
                parameters=parameters,
                hyperparameters=hyperparameters
            )
            st.pyplot(fig)
            st.table(error_df)


with tab_help:
    st.header("Work in progress... Here is kitty.")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)


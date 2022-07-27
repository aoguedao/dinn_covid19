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

    col1, col2, col3, col4 = st.columns(4)

    N = col1.number_input(
        "N",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        format=None,
        help="Population",
    )

    beta = col2.number_input(
        "Beta",
        min_value=0.001,
        max_value=1.0,
        value=0.5,
        step=0.001,
        format=None,
        help="Transmission Rate",
    )

    omega = col3.number_input(
        "Omega",
        min_value=0.001,
        max_value=1.0,
        value=1 / 14,
        step=0.001,
        format=None,
        help="Rate at which Infected individuals become Recovered",
    )

    gamma = col4.number_input(
        r"Gamma$",
        min_value=0.001,
        max_value=1.0,
        value=1 / 5,
        step=0.001,
        format=None,
        help="Rate at which Infected individuals become Dead",
    )

    if st.button("Run DINN"):
        with st.spinner('Wait for it...'):
            error_df, fig = run(N, beta, omega, gamma)
            st.pyplot(fig)
            st.table(error_df)


with tab_help:
    st.header("Work in progress... Here is kitty.")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)


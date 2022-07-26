import streamlit as st

from helpers import plot

st.set_page_config(
     page_title="SIRD with DINN",
     page_icon="🧊",
     layout="wide",
)

tab1, tab2 = st.tabs(["Model", "Help"])

with tab1:
    beta = 0.5
    omega = 1 / 14
    gamma = 1 / 5

    N = st.number_input(
        "N",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        format=None,
        key=None,
        help=None,
        on_change=None,
        disabled=False
    )

    st.write([N, beta, omega, gamma])

    st.latex(r'''
        a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
        \sum_{k=0}^{n-1} ar^k =
        a \left(\frac{1-r^{n}}{1-r}\right)
        ''')

    if st.button("Run"):
        fig = plot(N, beta, omega, gamma)
        st.pyplot(fig)


with tab2:
    st.header("ayudaaaaa")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)


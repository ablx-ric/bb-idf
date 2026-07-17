import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    from sklearn.preprocessing import add_dummy_feature

    X = 4*np.random.randn(100, 1)
    y = 5 + X + np.random.randn(100, 1)

    X_new = add_dummy_feature(X)

    X_new
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    \textbf{y} = \theta \times x
    $$
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

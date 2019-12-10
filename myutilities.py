from IPython.display import HTML
import random
import numpy as np
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import widgets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss


def compute_frequency(s):
    tot = s.sum(axis=0)
    f = s/tot
    return f


def compute_gini(f):
    # Gini index is defined as 1 - sum(rel_frequency^2)
    gini = 1 - pd.np.sum(f ** 2, axis=0)
    return gini


def compute_simpson(f):
    return pd.np.sum(f ** 2, axis=0)


def compute_shannon(f):
    return -pd.np.sum(f * np.log(f))


def evenness(s, H):
    # La evenness è una misura di diversità normalizzata su
    # una scala prefissata (es. da 0 a 1) e consente di
    # confrontare indici di diversità.

    Hmax = np.log(s)
    return np.divide(H, Hmax)


def index_compare(df1, df2, category):
    f1 = compute_frequency(df1)
    gini1 = compute_gini(f1)
    shannon1 = compute_shannon(f1)
    s1 = df1.nunique()
    gini_norm1 = np.divide(gini1 * s1, s1-1)
    J1 = evenness(s1, shannon1)

    f2 = compute_frequency(df2)
    gini2 = compute_gini(f2)
    shannon2 = compute_shannon(f2)
    s2 = df2.nunique()
    gini_norm2 = np.divide(gini2 * s2, s2-1)
    J2 = evenness(s2, shannon2)

    fig = go.Figure()

    layout = go.Layout(
        title="Index comparison between univariate and joint distribution",
        xaxis = dict(
            title = category + " + Target",
            showticklabels=False),
        yaxis = dict(
            title = 'Index value'
        )
    )
    fig.add_trace(go.Scatter(
        x=[0, 2],
        y=[gini_norm1, gini_norm2],
        name="Gini Index",
        mode = "lines+markers+text",
        text = ["Univariate","Joint"],
        textposition = "top right",
        hoverinfo="y+text"
    ))


    fig.add_trace(go.Scatter(
        x=[2.5, 4.5],
        y=[J1, J2],
        name="Shannon Index",
        mode = "lines+markers+text",
        text = ["Univariate","Joint"],
        textposition = "top right",
        hoverinfo="y+text"
    ))

    fig.update_layout(layout)

    fig.show()


def disproportion_index(df, category):
    f = compute_frequency(df)
    gini = compute_gini(f)
    shannon = compute_shannon(f)

    # get first column, all rows then compute number of unique values
    s = df.nunique()
    gini_norm = np.divide(gini * s, s-1)
    J = evenness(s, shannon)
    #print("Normalized Gini index: {:.2f}\nNormalized Shannon index: {:.2f}".format(gini_norm,J))

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Normalized Gini Index', 'Normalized Shannon Index'],
                    line_color='rgb(0,150,136)',
                    fill_color='rgb(0,150,136)',
                    align='left',
                    font=dict(color="white")),
        cells=dict(values=[[gini_norm.round(3)],  # 1st column
                           [J.round(3)]],  # 2nd column
                   line_color='rgb(0,150,136)',
                   fill_color='white',
                   align='left'))
    ])

    fig.update_layout(width=600, height=300,
                      title="Disproportion for" + category)
    fig.show()


def bar_plot(new_series, category):
    categoryorder = "array"
    if(category == "AGE-bin"):
        categoryorder = "category ascending"
    # new_series must have categorical attribute as index and number of occurrencies as value (in column)
    s = new_series
    perc = [str((100 * x).round(2)) + "%" for x in (s.array/s.sum())]

    fig = go.Figure(data=[go.Bar(
        x=s.index, y=s.array,
        text=perc,
        textposition='auto',
    )],
        layout=go.Layout(
        xaxis=dict(
            categoryorder=categoryorder
        )
    )
    )

    # Customize aspect
    fig.update_traces(marker_color='rgb(0,150,136)', marker_line_color='rgb(3,97,88)',
                      marker_line_width=1.5, opacity=0.8)
    title = "Frequencies for {} class".format(category)
    # for x in s.index:
    #     title += str(x) + ", "
    # title = str(title)[:-2]
    fig.update_layout(title_text=title)

    fig.show()


def heatmap_normal(x, y, z):
    f = go.Figure(data=go.Heatmap(
        xgap=1,
        ygap=1,
        name="Residual",
        hoverinfo="x+y+z+name",
        z=z,
        x=x,
        y=y,
        colorscale=greyscale(),
        colorbar=dict(
            tick0=0,
            dtick=1,
            title=dict(text="Pearson Residual",
                       font=dict(size=11), side="right")

        )
    )
    )
    return f


def chi2_pearson(column, df, target_column):
    showMargins = False
    sexXdefault = pd.crosstab(
        df[column], df[target_column], margins=showMargins)

    chi2, p, dof, expected = ss.chi2_contingency(sexXdefault)
    x = list(sexXdefault.columns.values)
    y = list(sexXdefault.index.values)
    psr = [x.round(3) for x in np.divide(
        sexXdefault.to_numpy() - expected, np.sqrt(expected))]
    #fig = heatmap_normal(x,y, psr)
    fig = fig = ff.create_annotated_heatmap(
        z=psr, x=x, y=y, colorscale=greyscale())
    fig.update_layout(
        title={
            'text': "Pearson Residuals For {} Attribute".format(column),
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},

    )
    fig.show()


def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    # bit of JS to permanently hide code in current cell (only when toggling next cell)
    js_hide_current = ''

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1, 2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current,
        toggle_text=toggle_text
    )

    return HTML(html)


def greyscale():
    g = [
        # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
        [0, "rgb(0, 0, 0)"],
        [0.1, "rgb(0, 0, 0)"],

        # # Let values between 10-20% of the min and max of z
        # # have color rgb(20, 20, 20)
        [0.1, "rgb(20, 20, 20)"],
        [0.2, "rgb(20, 20, 20)"],

        # # Values between 20-30% of the min and max of z
        # # have color rgb(40, 40, 40)
        [0.2, "rgb(40, 40, 40)"],
        [0.3, "rgb(40, 40, 40)"],

        [0.3, "rgb(60, 60, 60)"],
        [0.4, "rgb(60, 60, 60)"],

        [0.4, "rgb(80, 80, 80)"],
        [0.5, "rgb(80, 80, 80)"],

        [0.5, "rgb(100, 100, 100)"],
        [0.6, "rgb(100, 100, 100)"],

        [0.6, "rgb(120, 120, 120)"],
        [0.7, "rgb(120, 120, 120)"],

        [0.7, "rgb(140, 140, 140)"],
        [0.8, "rgb(140, 140, 140)"],

        [0.8, "rgb(160, 160, 160)"],
        [0.9, "rgb(160, 160, 160)"],

        [0.9, "rgb(180, 180, 180)"],
        [1.0, "rgb(180, 180, 180)"]
    ]
    return g

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly as py
from ipywidgets import widgets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as ss


def compute_frequency(df):
    total_sample = df['Count'].sum(axis=0) 
    f = df['Count']/total_sample
    f.name = 'frequencies'
    return f

def compute_gini(f):
    # Gini index is defined as 1 - sum(frequency^2)
    gini=  1 - pd.np.sum( f **2, axis=0)
    return gini

def compute_shannon(f):
    return -pd.np.sum(f * np.log(f))

def evenness(s,H):
    # La evenness è una misura di diversità normalizzata su
    # una scala prefissata (es. da 0 a 1) e consente di
    # confrontare indici di diversità.
    
    Hmax = np.log(s)
    return np.divide(H,Hmax)
    
def disproportion_index(df, category):
    f = compute_frequency(df)
    gini = compute_gini(f)
    shannon = compute_shannon(f)
    
    # get first column, all rows then compute number of unique values
    s = df.iloc[:,0].nunique()
    J = evenness(s,shannon)
    #print("Gini index: {:.2f}\nNormalized Shannon index: {:.2f}".format(gini,J))
    
    fig = go.Figure(data=[go.Table(
    header=dict(values=['Gini Index', 'Shannon Index'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[[gini.round(3)], # 1st column
                       [J.round(3)]], # 2nd column
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
    ])

    fig.update_layout(width=500, height=300, title = "Indici Di Disproporzione Per " + category)
    fig.show()
    
def bar_plot(new_series, category):
    # new_series must have categorical attribute as index and number of occurrencies as value (in column)
    s = new_series
    perc = [str((100 * x).round(2)) + "%" for x in (s.array/s.sum())]
    
    fig = go.Figure(data=[go.Bar(
            x=s.index, y=s.array,
            text=perc,
            textposition='auto',
        )])
    
    # Customize aspect
    fig.update_traces(marker_color='rgb(35, 207, 164)', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    title = "Categoria: " + category + " - Frequenza degli attributi: "
    for x in s.index:
        title += str(x) + ", "
    title = str(title)[:-2]
    fig.update_layout(title_text=title)

    fig.show()
    
def chi2_pearson(column, df, target_column):
    showMargins = False
    sexXdefault = pd.crosstab(df[column], df[target_column], margins = showMargins)
    
    chi2, p, dof, expected = ss.chi2_contingency(sexXdefault)

    psr = np.divide(sexXdefault.to_numpy() - expected, np.sqrt(expected))
    fig = go.Figure(data=go.Heatmap(
                        z = psr,
                        x = (sexXdefault.columns),
                        y = (sexXdefault.index)
                                    ) 
                   )
    
    fig.update_layout(
    title={
        'text': "Pearson Residuals For {} Attribute".format(column),
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    
    )
    fig.show()

from IPython.display import HTML
import random

def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

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
"""
stoclust.visualization

Contains functions for visualizing data and clusters.

Functions
---------
heatmap(mat,show_x=None,show_y=None,xlabels=None,ylabels=None,layout=None,**kwargs):

    Generates a heatmap of a given matrix: 
    that is, displays the matrix as a table of colored blocks 
    such that the colors correspond to matrix values.


scatter3D(x,y,z,agg=None,layout=None,show_items=None,**kwargs):

    Generates a 3-dimensional scatter plot 
    of given coordinate vectors; optionally plots 
    them on separate traces based on an aggregation.


scatter2D(x,y,agg=None,layout=None,show_items=None,**kwargs):

    Generates a 2-dimensional scatter plot 
    of given coordinate vectors; optionally plots 
    them on separate traces based on an aggregation.

bars(mat,show_x=None,show_y=None,xlabels=None,ylabels=None,layout=None,**kwargs):

    Generates a stacked bar plot of a given array of vectors; 
    the rows index the horizontally separate bars 
    and the columns index the stack heights.

dendrogram(hier,line=None,layout=None,show_progress=False,**kwargs):

    Generates a dendrogram of a hierarchical clustering scheme 
    in a Plotly Figure. Uses Plotly Shapes to draw 
    the dendrogram and a scatter plot to 
    highlight clusters at their branching points.

"""

import numpy as _np
import plotly.graph_objects as _go
from stoclust.Aggregation import Aggregation as _Aggregation
from stoclust.Group import Group as _Group
from tqdm import tqdm as _tqdm

def heatmap(mat,show_x=None,show_y=None,xlabels=None,ylabels=None,layout=None,**kwargs):
    """
    Generates a heatmap of a given matrix: that is, displays the matrix as a table of colored blocks such that the colors correspond to matrix values.

    Arguments
    ---------
    mat :       The matrix whose values are being visualized in a heatmap.

    Keyword Arguments
    -----------------
    show_x :    An array of the column indices which are to be shown, in the order they should be shown.

    show_y :    An array of the row indices which are to be shown, in the order they should be shown.

    xlabels :   An array or group of how the columns should be labeled on the plot.

    ylabels :   An array or group of how the rows should be labeled on the plot.

    layout :    A dictionary for updating values for the Plotly Figure layout.

    **kwargs :  Keyword arguments for the Plotly Heatmap trace.

    Output
    ------
    fig :       A Plotly Figure containing the heatmap.
    """
    if show_x is None:
        show_x = _np.arange(mat.shape[1])
    if show_y is None:
        show_y = _np.arange(mat.shape[0])
    if xlabels is None:
        xlabels = _np.arange(mat.shape[1])
    if ylabels is None:
        ylabels = _np.arange(mat.shape[0])

    fig = _go.Figure(data=_go.Heatmap(
                        z=mat[show_y][:,show_x],**kwargs))
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = _np.arange(len(show_x)),
            ticktext = xlabels[show_x]
        ),
        yaxis = dict(
            tickmode = 'array',
            tickvals = _np.arange(len(show_y)),
            ticktext = ylabels[show_y]
        ),
        margin=dict(l=100, r=100, t=20, b=20),
    )

    if layout is not None:
        fig.update_layout(**layout)

    return fig

def scatter3D(x,y,z,agg=None,layout=None,show_items=None,**kwargs):
    """
    Generates a 3-dimensional scatter plot of given coordinate vectors; optionally plots them on separate traces based on an aggregation.

    Arguments
    ---------
    x :             The x-coordinates of the data points.

    y :             The y-coordinates of the data points.

    z :             The z-coordinates of the data points.

    Keyword Arguments
    -----------------
    agg :           An Aggregation of the indices of x, y and z.

    show_items :    A one-dimensional array of which indices of x, y and z are to be shown.

    layout :        A dictionary for updating values for the Plotly Figure layout.

    **kwargs :      Keyword arguments for the Plotly Scatter3d trace.
                    If an attribute is given as a single string or float, will be applied to all data points. 
                    If as an array of length x.shape[0], will be applied separately to each data point.
                    If an an array of length agg.clusters.size, will be applied separately to each cluster.


    Output
    ------
    fig :           A Plotly Figure containing the scatter plot.
    """
    if agg is None:
        agg = _Aggregation(_Group(_np.arange(x.shape[0])),
                          _Group(_np.array([0])),
                          {0:_np.arange(x.shape[0])})
    specific_keywords = [{} for i in range(agg.clusters.size)]
    for k,v in kwargs.items():
        if hasattr(v, '__len__') and not(isinstance(v,str)):
            if len(v)==len(agg.clusters):
                for i in range(agg.clusters.size):
                    specific_keywords[i][k] = v[i]
            elif len(v)==len(agg.items):
                for i in range(agg.clusters.size):
                    specific_keywords[i][k] = v[agg._aggregations[i]]
        else:
            for i in range(agg.clusters.size):
                specific_keywords[i][k] = v
    if kwargs.get('name',None) is None:
        for i in range(agg.clusters.size):
            specific_keywords[i]['name'] = str(agg.clusters.elements[i])
    fig = _go.Figure(data=[_go.Scatter3d(x=x[agg._aggregations[i]],
                                       y=y[agg._aggregations[i]],
                                       z=z[agg._aggregations[i]],
                                       **(specific_keywords[i]))                   
                        for i in range(agg.clusters.size)])
    if layout is not None:
        fig.update_layout(**layout)
    return fig

def scatter2D(x,y,agg=None,layout=None,show_items=None,**kwargs):
    """
    Generates a 2-dimensional scatter plot of given coordinate vectors; optionally plots them on separate traces based on an aggregation.

    Arguments
    ---------
    x :             The x-coordinates of the data points.

    y :             The y-coordinates of the data points.

    Keyword Arguments
    -----------------
    agg :           An Aggregation of the indices of x and y.

    show_items :    A one-dimensional array of which indices of x and y are to be shown.

    layout :        A dictionary for updating values for the Plotly Figure layout.

    **kwargs :      Keyword arguments for the Plotly Scatter trace.
                    If an attribute is given as a single string or float, will be applied to all data points. 
                    If as an array of length x.shape[0], will be applied separately to each data point.
                    If an an array of length agg.clusters.size, will be applied separately to each cluster.


    Output
    ------
    fig :           A Plotly Figure containing the scatter plot.
    """
    if agg is None:
        agg = _Aggregation(_Group(_np.arange(x.shape[0])),
                          _Group(_np.array([0])),
                          {0:_np.arange(x.shape[0])})


    specific_keywords = [{} for i in range(agg.clusters.size)]
    for k,v in kwargs.items():
        if hasattr(v, '__len__') and not(isinstance(v,str)):
            if len(v)==len(agg.clusters):
                for i in range(agg.clusters.size):
                    specific_keywords[i][k] = v[i]
            elif len(v)==len(agg.items):
                for i in range(agg.clusters.size):
                    specific_keywords[i][k] = v[agg._aggregations[i]]
        else:
            for i in range(agg.clusters.size):
                specific_keywords[i][k] = v
    if kwargs.get('name',None) is None:
        for i in range(agg.clusters.size):
            specific_keywords[i]['name'] = str(agg.clusters.elements[i])

    fig = _go.Figure(data=[_go.Scatter(x=x[agg._aggregations[i]],
                                       y=y[agg._aggregations[i]],
                                       **(specific_keywords[i]))                   
                        for i in range(agg.clusters.size)])
    if layout is not None:
        fig.update_layout(**layout)
    return fig

def bars(mat,show_x=None,show_y=None,xlabels=None,ylabels=None,layout=None,**kwargs):
    """
    Generates a stacked bar plot of a given array of vectors; the rows index the horizontally separate bars and the columns index the stack heights.

    Arguments
    ---------
    mat :       The matrix whose values are being visualized in a stacked bar plot.

    Keyword Arguments
    -----------------
    show_x :    An array of the row indices (horizontally separate bars) which are to be shown, in the order they should be shown.

    show_y :    An array of the column indices (stacked bars) which are to be shown, in the order they should be shown.

    xlabels :   An array or group of how the rows should be labeled on the plot.

    ylabels :   An array or group of how the columns should be labeled on the plot.

    layout :    A dictionary for updating values for the Plotly Figure layout.

    **kwargs :  Keyword arguments for the Plotly Bar trace. 
                If an attribute is given as a single string or float, will be applied to all bars. 
                If as an array of length mat.shape[1], will be applied separately to each layer of the stack.

    Output
    ------
    fig :       A Plotly Figure containing the stacked bars.
    """
    if show_x is None:
        show_x = _np.arange(mat.shape[0])
    if show_y is None:
        show_y = _np.arange(mat.shape[1])
    if xlabels is None:
        xlabels = _np.arange(mat.shape[0]).astype(str)
    if ylabels is None:
        ylabels = _np.arange(mat.shape[1]).astype(str)

    specific_keywords = [{} for i in range(mat.shape[1])]
    for k,v in kwargs.items():
        if hasattr(v, '__len__') and not(isinstance(v,str)):
            if isinstance(v,_np.ndarray):
                if len(v.shape)==2:
                    for i in range(mat.shape[1]):
                        specific_keywords[i][k] = v[k,i]
                else:
                    for i in range(mat.shape[1]):
                        specific_keywords[i][k] = v[i]
            else:
                for i in range(mat.shape[1]):
                    specific_keywords[i][k] = v[i]
        else:
            for i in range(mat.shape[1]):
                specific_keywords[i][k] = v

    if kwargs.get('width',None) is None:
        for i in range(mat.shape[1]):
            specific_keywords[i]['width'] = 1 

    fig = _go.Figure(data=[
        _go.Bar(name=ylabels[o], x=xlabels, y=mat[show_x,o], **specific_keywords[o]) for o in show_y
    ])
    fig.update_layout(barmode='stack',
                    xaxis = dict(
                        tickmode = 'array',
                        tickvals = _np.arange(len(show_x)),
                        ticktext = (xlabels)[show_x]
                    ),)

    if layout is not None:
        fig.update_layout(**layout)
    return fig


def dendrogram(hier,line=None,layout=None,show_progress=False,**kwargs):
    """
    Generates a dendrogram of a hierarchical clustering scheme in a Plotly Figure. Uses Plotly Shapes to draw the dendrogram and a scatter plot to highlight clusters at their branching points.

    Arguments
    ---------
    hier :          A Hierarchy which is to be plotted as a Dendrogram.

    Keyword Arguments
    -----------------
    line :          A dict for formatting Plotly shape lines.
                    If an attribute is given as a single string or float, will be applied to all lines.
                    If as an array of length hier.clusters.size, will be applied separately to the lines immediately beneath each cluster.

    layout :        A dictionary for updating values for the Plotly Figure layout.

    show_progress : Boolean; whether to show a tqdm progress bar as the dendrogram is generated.

    **kwargs :      Keyword arguments for the Plotly Scatter trace. 
                    If an attribute is given as a single string or float, will be applied to all branch points. 
                    If as an array of length hier.clusters.size, will be applied separately to each cluster's branch point.

    Output
    ------
    fig :           A Plotly Figure containing the dendrogram.
    """
    groups = hier.cluster_groups()

    x_items = _np.zeros([hier.items.size])
    s_max = _np.max(hier._scales)
    top_agg = hier.at_scale(s_max)
    x_base = 0
    x_in_superset = []
    for c in range(top_agg.clusters.size):
        grp = top_agg._aggregations[c]
        n = len(grp)
        x_items[grp] = _np.arange(n)+x_base
        x_base += n
        x_in_superset = x_in_superset + list(top_agg._aggregations[c])
    x_in_superset = _np.array(x_in_superset)
    
    x_clusters = _np.zeros([hier.clusters.size])
    y_clusters = _np.zeros([hier.clusters.size])
    fig = _go.Figure()

    lineinfo = [{} for c in range(hier.clusters.size)]
    if line is None:
        for c in range(hier.clusters.size):
            lineinfo[c]=dict(
                    color="RoyalBlue",
                    width=3)
    else:
        for k,v in line.items():
            if hasattr(v, '__len__') and not(isinstance(v,str)):
                for c in range(hier.clusters.size):
                    lineinfo[c][k] = v[c]
            else:
                for c in range(hier.clusters.size):
                    lineinfo[c][k] = v
    if show_progress:
        clust_iter = _tqdm(range(hier.clusters.size))
    else:
        clust_iter = range(hier.clusters.size)
    
    for c in clust_iter:
        x_clusters[c] = _np.average(x_items[groups[hier.clusters[c]].in_superset])
        y_clusters[c] = hier._scales[c]
        if len(hier._children[c])>0:
            xmin = _np.min(x_clusters[hier._children[c]])
            xmax = _np.max(x_clusters[hier._children[c]])
            fig.add_shape(
                        # Line Horizontal
                        dict(
                            type="line",
                            x0=xmin,
                            y0=y_clusters[c],
                            x1=xmax,
                            y1=y_clusters[c],
                            line=lineinfo[c]
                    ))
            for k in hier._children[c]:
                fig.add_shape(
                            # Line Vertical
                            dict(
                                type="line",
                                x0=x_clusters[k],
                                y0=y_clusters[k],
                                x1=x_clusters[k],
                                y1=y_clusters[c],
                                line=lineinfo[c]
                        ))
        
    if kwargs.get('customdata',None) is None:
        customdata=hier.clusters.elements
    if kwargs.get('hovertemplate',None) is None:
        hovertemplate = '<b>ID</b>: %{customdata} <br><b>Scale</b>: %{y} '
    fig.add_trace(_go.Scatter(x=x_clusters,y=y_clusters,
                    mode='markers',
                    customdata=customdata, 
                    hovertemplate = hovertemplate,**kwargs))
    fig.update_layout(
            title = kwargs.get('title','Dendrogram'),
            margin=dict(l=20, r=20, t=30, b=10),
            xaxis_title=kwargs.get('x_axis_label','Items'),
            yaxis_title=kwargs.get('y_axis_label','Scale'),
            xaxis = dict(
                tickmode = 'array',
                tickvals = _np.arange(hier.items.size),
                ticktext = hier.items.elements[x_in_superset]
            ))
    fig.update_shapes(layer='below')
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(showgrid=False,zeroline=False)
    if layout is not None:
        fig.update_layout(layout)
    return fig

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# data is dataset selected error metric with varying alg and single param
def plot_cca_vs_c3a_over_parameter(data, param, ylabel, title, log_scale=True):
    '''
    Plot multiple algorithms over a given parameter on one plot of error metric vs. samples/feature.
    Parameters:
    -------------
    data: xarray.DataArray
        data for selected error metric (e.g. 'cmntrth_weight_error') over varying 'alg' and param 
    param: string
        param over which to plot multiple lines
    ylabel: string
    title: string
    log_scale: bool
        Whether to log scale both axes
    '''
    fig, axs = plt.subplots(1, 1, squeeze=False)

    ax = axs[0, 0]

    for index, n in enumerate(data[param].values):
        ax.plot(data.n_per_ftr1, data.sel({'alg':'c3a', param:n}), label="{}={}".format(param, n))
    if log_scale:
        ax.loglog(data.n_per_ftr1, data.sel({'alg':'cca', param: n}), label='c3a')
    else:
        ax.plot(data.n_per_ftr1, data.sel({'alg':'cca', param: n}), label='cca')
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_ylim(None, 1)
    ax.set_xlabel('Samples / feature')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax

def plot_algs(datasets, ylabel, title, log_scale=True):
    '''
    Plot all algs on one plot of error measure vs. samples/feature
    Parameters:
    -------------
    datasets: [xarray.DataArray]
        data for selected error metric (e.g. 'cmntrth_weight_error') over varying 'alg' 
    ylabel: string
    title: string
    log_scale: bool
        Whether to log scale both axes
    
    '''
    fig, axs = plt.subplots(1, 1, squeeze=False)

    ax = axs[0, 0]
    for data in datasets:
        for alg in data.alg.values:
            if log_scale:
                ax.loglog(data.n_per_ftr1, data.sel(alg=alg), label=alg)
            else:
                ax.plot(data.n_per_ftr1, data.sel(alg=alg), label=alg)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_ylim(None, 1)
    ax.set_xlabel('Samples / feature')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def calc_samples_for_error_threshold(x, data, threshold):
    return interpolate_zero(np.abs(data.values) - threshold, x)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_subplots_nx2_grid(dataset, param,n=2):
    '''
    Plots C3A and CCA on each plot in an nx2 grid. 
    Parameters:
    -------------
    dataset: xarray.DataArray
        data for selected error metric (e.g. 'cmntrth_weight_error') over varying 'alg' and param 
    param: string
        param over which to plot multiple lines
    n: int
        number of rows of plots in the figure
    '''
    fig, ax = plt.subplots(n, 2, sharex='col', sharey='row', figsize=(7, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for index, x in enumerate(dataset[param].values):
        x_index = index %2
        y_index = int(index/n)
        ax[x_index, y_index].loglog(dataset.n_per_ftr1, dataset.sel({'alg':'cca', param:x}), linestyle='--', c=color)
        ax[x_index, y_index].loglog(dataset.n_per_ftr1, dataset.sel({'alg':'c3a', param:x}))
        ax[x_index, y_index].set_ylim(None, 1)
        ax[x_index, y_index].set_title("{}={}".format(param, x))
    return fig, ax

def interpolate_zeros_over_parameter(x, y, param_name, alg_names=('cca', 'c3a')):
    """ Find first point at which CCA outperforms C3A, e.g CCA error - C3A error < 0
    If CCA always outperforms C3A for the value of the param, append 0. 
    If C3A always outperforms CCA, append NaN.

    Parameters
    ----------
    x: ndarray (len(# n_per_ftr1 values))
    y: xarray.Dataset with coordinates 'cca' and 'c3a

    Returns:
    zeros: ndarray of length len(y[param_name].values)
    """
    zeros = []
    for n in y[param_name].values:
        if n == 'cca':
            continue
        y_n = y.sel({param_name: n})
        if param_name=='alg':
            delta_y= y.sel(alg='cca').values - y_n.values
        else:
            delta_y = y_n.sel(alg=alg_names[0]).values - y_n.sel(alg=alg_names[1]).values
        zeros.append(interpolate_zero(delta_y, x))

    return zeros

def interpolate_zero(data, x):
    '''
    Interpolate x-value where data goes from positive to negative. 
    Returns: int
    '''
    index = find_sign_change(data)
    if index == -1:
        return 0
    elif index == -2:
        return np.NaN
    else:
        return interpolate(x.values, data, index)

def find_sign_change(y):
    '''
    Find index in array where between index and index +1, y values go from positive to negative.

    Returns: int
    '''
    for i in range(len(y)):
        if y[i] == 0:
            return i
        elif y[i] < 0 and y[i+1] < 0:
            return -1;
        elif i == len(y) -1:
            return -2
        elif y[i] > 0 and y[i+1] < 0:
            return i

def interpolate(x, y, index):
    '''
    Given an index in the array, interpolate the x-value between the index
     and the next index in the array where a zero occurs.

    Parameters:
    ----------
    x: ndarray of length n_per_ftr1
        X values of n_per_ftr1
    y: ndarray of length n_per_ftr1
        Y values corresponding to X values
    index: int
        where Y values cross zero between index and index+1

    Returns: int
    '''
    if index == -1: 
        raise ValueError('Index out of bounds')
    if y[index] == 0:
        return x[index]
    slope = (y[index +1] - y[index])/(x[index+1] - x[index])
    return x[index] - y[index]/slope

def get_interpolation_matrix(x, y_values, row_param, col_param):
    '''
    Get data for heatmap of required samples/feature for CCA to outperform C3A. Note: if plotting 
    r on one dimension, pass 'rxa' to row_param.

    Parameters: 
    ------------
    x: ndarray of length # n_per_ftr1
        n_per_ftr1 values
    y_values: xarray.DataArray
        Data for some error metric (e.g. 'cmntrth_weight_error') over row_param and col_param
    row_param: string
        parameter for rows of heatmap
    col_param: string
        parameter for cols of heatmap
    
    Returns:
    data: ndarray (len(y_values[row_param]), len(y_values[col_param]))
    '''
    data = np.empty((len(y_values[row_param].values),len(y_values[col_param].values)), float)
    for index, value in enumerate(y_values[row_param].values):
        if row_param=='rxa':
            y_sel = y_values.sel({'rxa': value, 'rxb':value})
        else:
            y_sel = y_values.sel({row_param: value})
        required_n = interpolate_zeros_over_parameter(x, y_sel, col_param)
        data[index] = required_n
    return data

def get_required_samples_matrix(x, y_values, row_param, col_param):
    '''
    Get data for heatmap of required samples/feature for alg to reach error threshold. Note: if plotting 
    r on one dimension, pass 'rxa' to row_param.

    Parameters: 
    ------------
    x: ndarray of length # n_per_ftr1
        n_per_ftr1 values
    y_values: xarray.DataArray
        Data for some error metric (e.g. 'cmntrth_weight_error') over row_param and col_param
    row_param: string
        parameter for rows of heatmap
    col_param: string
        parameter for cols of heatmap
    
    Returns:
    data: ndarray (len(y_values[row_param]), len(y_values[col_param]))
    '''
    data = np.empty((len(y_values[row_param].values),len(y_values[col_param].values)), float)
    for index, value in enumerate(y_values[row_param].values):
        if row_param=='rxa':
            y_sel = y_values.sel({'rxa': value, 'rxb':value})
        else:
            y_sel = y_values.sel({row_param: value})
        required_n = [interpolate_zero(y_sel.sel({col_param: n}).values, x) for n in y_sel[col_param].values]
        data[index] = required_n
    return data


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
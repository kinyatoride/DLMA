import numpy as np
from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import warnings

def plot_histpdf(xvar, yvar, **kwargs):

    """
    Function to create and plot smoothed probability density functions
    for 2-dimensional histograms

    Input:
        xvar [numpy array] - input vector for x
        yvar [numpy array] - input vector for y

        Optional:
            fig : Matplotlib Figure instance
                Figure where to plot the histpdfs.
                If not given, a new plot will be opened.
            ax  : AxesSubplot instance
                where to plot the histpdfs
            levels : list or int
                probabilities for contours
                Default = [0.5, 0.9]
            xscale : str
                define scale of x-axis
                in case of logarithmic scales (e.g. for H2O), it is recommended to set
                xscale = 'log'
                Default is ''
            yscale : str
                define scale of x-axis
                Default is ''
            lw  : int
                linewidth for contours.
                Default is 1
            ls  : str or list of strings
                linestyle
                if str, then the same linestyle will be applied to all contours
                if list, it must have the same length as levels and each
                         line plot will have an individual given linestyle
                Default is None (i.e. 'solid')
            xbins : int
                number of bins considered for the x axis
                default is 50
            ybins : int
                number of bins considered for the y axis
                default is 100
            xlim  : list
                the shape [xmin, xmax] is expected to define the x axis limits
                for creating the 2d histogram
            ylim  : list
                the shape [ymin, ymax] is expected to define the y axis limits
                for creating the 2d histogram
            label : str
                label for line plot, may be used for legend drawing
                default is None
            color : str
                Color for contour lines.
                Default is black
            gauss_val : int
                Gauss value for gaussian filtering
                Default is 15
            l_interp_center : bool
                Switch for centering bins
                Default = True
            l_gaussian_filter : bool
                Switch for using gaussian smoothing filter
                Default = True
            l_add_scatter : bool
                Switch to add underlying scatter
                Default = False
            color_scatter : str
                Color for scatter
                Default = darkgray
    Return:
        cs_contour_list
            may be used for drawing legends
    """
    # check keyword arguments
    fig               = kwargs.get('fig')
    ax                = kwargs.get('ax')

    lw                = kwargs.get('lw', 1.)
    ls                = kwargs.get('ls')
    color             = kwargs.get('color', 'k')
    levels             = kwargs.get('levels', [0.5, 0.9])

    gauss_val         = kwargs.get('gauss_val', 15)

    l_interp_center   = kwargs.get('l_interp_center', True)
    l_gaussian_filter = kwargs.get('l_gaussian_filter', True)

    l_add_scatter     = kwargs.get('l_add_scatter', False)
    color_scatter     = kwargs.get('color_scatter', 'darkgray')

    xscale            = kwargs.get('xscale', '')
    yscale            = kwargs.get('yscale', '')
    xbins             = kwargs.get('xbins', 50)
    ybins             = kwargs.get('ybins', 50)
    xlim              = kwargs.get('xlim')
    ylim              = kwargs.get('ylim')

    label             = kwargs.get('label')

    l_add_center      = kwargs.get('l_add_center', False)

    # Convert into list
    if not isinstance(levels, (list, np.ndarray)):
        levels = [levels]
    if not isinstance(ls, (list, np.ndarray)):
        ls = [ls]

    # open figure
    if fig is None:
        plt.figure()

    if ax is None:
        ax = plt.gca()

    # check for nans
    mask = ~np.isnan(xvar) & ~np.isnan(yvar)

    if all(~mask):
        warnings.warn('get_histpdf: Variables only consist of NaNs.')
        return None, ''
    else:
        xvar = xvar[mask]
        yvar = yvar[mask]

    # create bins
    if xlim is None:
        xmin = xvar.min() - 0.1 * (xvar.max() - xvar.min())
        xmax = xvar.max() + 0.1 * (xvar.max() - xvar.min())
        if xscale == 'log':
            xmin = np.exp(np.log(xvar.min()) - 0.1 * (np.log(xvar.max()) - np.log(xvar.min())))
            xmax = np.exp(np.log(xvar.max()) + 0.1 * (np.log(xvar.max()) - np.log(xvar.min())))
    else:
        xmin, xmax = xlim

    if ylim is None:
        ymin = yvar.min() - 0.1 * (yvar.max() - yvar.min())
        ymax = yvar.max() + 0.1 * (yvar.max() - yvar.min())
        if yscale == 'log':
            ymin = np.exp(np.log(yvar.min()) - 0.1 * (np.log(yvar.max()) - np.log(yvar.min())))
            ymax = np.exp(np.log(yvar.max()) + 0.1 * (np.log(yvar.max()) - np.log(yvar.min())))
    else:
        ymin, ymax = ylim

    if xscale == 'log':
        xbins = np.logspace(np.log10(xmin), np.log10(xmax), xbins)
    else:
        xbins = np.linspace(xmin, xmax, xbins)

    if yscale == 'log':
        ybins = np.logspace(np.log10(ymin), np.log10(ymax), ybins)
    else:
        ybins = np.linspace(ymin, ymax, ybins)

    # create 2 dim histogram
    hist, xedges, yedges  = np.histogram2d(xvar, yvar, bins=[xbins, ybins])
    hist = hist.T
    n_points = int(np.sum(hist))

    # to 1D
    hist1d = hist.ravel()

    # sort 1d histogram
    ind = np.argsort(hist1d)[::-1]
    hist1d_sort = hist1d[ind]

    # cumulative sum
    hist1d_sort_cum = hist1d_sort.cumsum()

    # Return to the original array
    ind_back = np.arange(len(hist1d))[np.argsort(ind)]
    hist_cum = hist1d_sort_cum[ind_back].reshape(hist.shape)

    # normalize
    hist = hist_cum / np.max(hist_cum)

    # find centers of bins for correct plotting
    xcenters = 0.5 * (xedges[1:] + xedges[:-1])
    ycenters = 0.5 * (yedges[1:] + yedges[:-1])

    # interpolate to finer grid
    if l_interp_center:
        if xscale == 'log':
            xcenters_itp = np.logspace(np.log10(xcenters.min()), np.log10(xcenters.max()), 1000)
        else:
            xcenters_itp = np.linspace(xcenters.min(), xcenters.max(), 1000)
        if yscale == 'log':
            ycenters_itp = np.logspace(np.log10(ycenters.min()), np.log10(ycenters.max()), 1000)
        else:
            ycenters_itp = np.linspace(ycenters.min(), ycenters.max(), 1000)
        hist = interpolate.interp2d(xcenters, ycenters, hist, kind='cubic')(xcenters_itp, ycenters_itp)

        xcenters = xcenters_itp
        ycenters = ycenters_itp

    # apply gaussian smoothing
    if l_gaussian_filter:
        hist = ndimage.gaussian_filter(hist, gauss_val)

    # add scatter points
    if l_add_scatter:
        yscatter = yvar
        xscatter = xvar
        cs_scatter = ax.scatter(xscatter, yscatter, alpha=0.3, color=color_scatter)

    if l_add_center:
        ix, iy = np.where(hist == np.min(hist))
        ax.plot(xcenters[iy], ycenters[ix], 'x', color = color)
        print('Center at: ', xcenters[iy], ycenters[ix])

    def fmt(x):
        return f'{x*100:.0f}%'

    cs_contour_list = []
    # plot
    for idx, level in enumerate(levels):
        c_lw = 4 * ((idx + 1) / (idx + 4)) * lw

        if len(ls) == 1:
            idx = 0

        cs_contour = ax.contour(xcenters, ycenters, hist,
                                levels=[level], linewidths=c_lw,
                                colors=color,
                                linestyles=ls[idx],
                                label=f'{label} {level*100:.0f}%',
                                )
        # ax.clabel(cs_contour, inline=True, inline_spacing=0, fontsize=10)
        cs_contour_list.append(cs_contour)

    return cs_contour_list
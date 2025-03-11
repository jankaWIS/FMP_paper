# import os
# import time

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def label_correlation(x, y, ax, xy=(.08, .85), plot_corr=True, **kwargs):
    # TODO - validate and test the change with kwargs
    """
    Label plot with correlation, returns the correlation values and the p-value
    Parameters
    ----------
    x: array, values to be correlated
    y: array, values to be correlated
    ax: matplotlib axis to be annotated
    xy: tuple of floats <0,1>, position of the label of the correlation
    plot_corr: bool, default True, whether to plot the correlation

    Returns
    -------
    r, p -- correlation coefficient and p-value

    """
    # find nans
    nans = np.logical_or(np.isnan(x), np.isnan(y))
    # add correlation label
    r, p = stats.pearsonr(x[~nans], y[~nans])
    if plot_corr:
        ax = ax or plt.gca()
        ax.annotate(f"r = {r:.2f}\np = {p:.2e}", xy=xy, xycoords=ax.transAxes, **kwargs)
    return r, p


def label_column_height(ax, x=0.35, y=0.03, size=12, rot=0, n=2):
    """
    adds a text label of how height the bar is
    x: float, vertical position of the label
    y: float, how much above the bar the label should be
    size: int, fontsize
    rot: int, rotation
    n: int, number of decimal places to show
    """
    for i, bar in enumerate(ax.patches):
        h = bar.get_height()
        # I don't know since when but in April 2024, new patches are added, seems like legend,
        # and these also get labelled. To avoid that, I'm checking for their label and heigh. TODO investigate
        if bar.get_label() != '_nolegend_' or h==0:
            print(f'This patch could not be labelled, h={h}, {bar.get_label()}, x={bar.get_x()}')
            continue
        # control for negative labelling
        if h < 0:
            height = h - y
        else:
            height = h + y
        ax.text(
            #             i, # bar index (x coordinate of text)
            bar.get_x() + x,  # bar index (x coordinate of text)
            height,  # y coordinate of text
            str('{:.' + str(n) + 'f}').format(h),  # y label
            ha='center',
            va='center',
            #         fontweight='bold',
            rotation=rot,
            size=size)


def change_width(ax, new_value):
    """
    Set the width nicely, see https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
    """
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def star_gazing(p):
    """
    Given a p-value, return a string of stars corresponding to significance.
    '' for p>= 0,5
    *  for 0,1 >=p< 0,5
    ** for 0,01>=p< 0,1
    *** for p< 0,001
    """

    if p < 10 ** (-3):
        return '***'
    elif p < 10 ** (-2):
        return '**'
    elif p < 5 * 10 ** (-2):
        return '*'
    return ''


def sharey_ax(a, source):
    """
    Takes axis a that we want to share with axis source, for CanD
    """
    # sharey, take the first axis and sharey with the last plotted one
    a.sharey(source)
    # need to rescale all axis
    a.autoscale()
    # remove ticks from shared y axes
    plt.setp(a.get_yticklabels(), visible=False)
    # remove ylabel
    a.set_ylabel('')


def sharey_name(a, source, c):
    """
    Takes name a of an axis that we want to share with an axis named source, need CanD canvas c, for CanD
    """
    # sharey, take the first axis and sharey with the last plotted one
    c.ax(a).sharey(c.ax(source))
    # need to rescale all axis
    c.ax(a).autoscale()
    # remove ticks from shared y axes
    plt.setp(c.ax(a).get_yticklabels(), visible=False)
    # remove ylabel
    c.ax(a).set_ylabel('')


def map_rev_palette(input_palette):
    """
    Go from the ordered palette to one that goes by tasks and difficulties and reverses the order of colours
    """
    # The hardcoded index mapping based on the relationship observed
    if len(input_palette) == 9:
        index_mapping = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    elif len(input_palette) == 12:
        index_mapping = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    else:
        raise ValueError('Not defined mapping.')

    # Ensure the input list is the correct length
    if len(input_palette) != len(index_mapping):
        raise ValueError("Input palette must have exactly 9 or 12 elements.")

    # Apply the index mapping
    mapped_palette = [input_palette[i] for i in index_mapping]

    return mapped_palette


def plot_FMP_results_slopes(data, ax, flattened_palette,
                            hue_order=['Matching', 'Unfilled Delay', 'Filled Delay - emotions'],
                            plot_legend=True, title='', legend_kwargs=None):
    """
    Plot results of the Face Matching Perception (FMP) task with slopes for different task conditions.

    This function generates a bar plot with overlaid individual data points and adds lines representing
    slopes for each task condition, indicating how performance varies with difficulty levels. It also
    customises the plot appearance and optionally adds a legend.

    Parameters
    ----------
    data: pandas.DataFrame, df containing the data to be plotted, with columns including 'difficulty',
        'correct_flt', 'task', and 'userID'. Each row represents an observation for a specific participant, task,
        and difficulty level.
    ax: matplotlib.axes.Axes, Matplotlib Axes object where the plot will be drawn.
    flattened_palette: list of str, list of colors to be used for plotting bars and points, ordered according to task
        conditions. Usually provided by get_flattened_palette(hue_order) call.
    hue_order: list of str, optional, list specifying the order of task conditions for plotting, with default values
        ['Matching', 'Unfilled Delay', 'Filled Delay - emotions']. Originally with the old naming it was
        hue_order=['face matching', 'delay', 'interference'].
    plot_legend: bool, optional, default True, if True, a legend will be displayed on the plot.
    title : str, optional, default is an empty string, title of the plot, displayed at the top.
    legend_kwargs: dict, optional, default None, additional keyword arguments for ax.legend().

    The function modifies the provided Axes object in-place, adding bars, points, lines, and other customisations.

    Notes
    -----
    - The function first plots bars representing mean performance across difficulty levels for each task condition,
      then overlays individual data points using a strip plot.
    - A dashed line indicating the chance level (0.5) is added.
    - Colours of bars and points are customised to match the provided palette.
    - Slopes are calculated and plotted for each task condition to illustrate performance trends.
    """
    # plot bars and add individual points
    bars = sns.barplot(x='difficulty', y='correct_flt', data=data,
                       hue='task', hue_order=hue_order, palette=flattened_palette, edgecolor='k',
                       ax=ax,
                       )

    stripplot = sns.stripplot(x='difficulty', y='correct_flt',
                              data=data.groupby(['userID', 'task', 'difficulty']).correct_flt.mean().reset_index(),
                              edgecolor='k', hue='task', dodge=True, hue_order=hue_order,  # color='white', alpha=0.2,
                              palette=flattened_palette,
                              ax=ax, size=3, linewidth=0.3, legend=False,
                              )

    # add chance level line
    ax.axhline(0.5, c='k', linestyle="--", label='chance level', zorder=4)

    # repaint the bars
    for bar, color in zip(bars.patches, flattened_palette):
        bar.set_facecolor(color)

    # get reversed colour names
    reordered_palette_rev = map_rev_palette(flattened_palette)

    # repaint the points
    for point, color in zip(stripplot.collections, reordered_palette_rev):
        point.set_facecolor(color)

    # legend
    if plot_legend:
        # deal with optional keyword arguments to the legend
        if legend_kwargs is None:
            # it can't deal with None and the default should not be mutable, so if it's none, convert it here to {}
            legend_kwargs = {}
        elif not isinstance(legend_kwargs, dict):
            raise TypeError("legend_kwargs must be a dictionary.")

        # get the legend
        legend = ax.legend()
        legend_handles = legend.legendHandles

        # Update the legend colors and labels
        new_labels = []
        for i, (handle, label) in enumerate(zip(legend_handles, hue_order)):
            handle.set_facecolor(flattened_palette[i * 3 + 1])
            new_labels.append(label.replace('delay', 'unfilled delay').title().replace(' - ', '\n'))

        # Regenerate the legend with updated labels on the axis level
        ax.legend(handles=legend_handles, labels=new_labels, title='Task', **legend_kwargs)
    else:
        ax.legend_.remove()

    # set bcg colour
    ax.set_facecolor((1, 1, 1, 0))

    # add slopes
    y = np.zeros(len(hue_order) * 3)
    x_coor = np.zeros(len(hue_order) * 3)
    for i, bar in enumerate(ax.patches):
        if bar.get_height() != 0:
            y[i] = bar.get_height()
            # record also position
            x_coor[i] = bar.get_x() + bar.get_width() / 2

    x_coor = x_coor.reshape((len(hue_order), 3))
    y = y.reshape((len(hue_order), 3))
    # define x array -> three levels of difficulty
    x = np.array([1, 2, 3])
    # compute the slopes
    for i in range(y.shape[0]):
        slope, intercept = np.polyfit(x, y[i], 1)

        # add a line to the graph, first create space that accounts for the shifted bars
        space = np.linspace(x_coor[i].min(), x_coor[i].max(), 50)
        # this is the actual prediction
        space1 = np.linspace(x.min(), x.max(), 50)
        ax.plot(space, space1 * slope + intercept, color=flattened_palette[i * 3], linewidth=2, zorder=2)

    # label axis and update limits
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.15, 1.05)
    ax.set_xlabel("Perceptual difficulty level")
    ax.set_title(f"{title}N={data.userID.unique().size}")

    sns.despine(ax=ax)


def get_flattened_palette(hue_order):
    """
    This function creates a colour palette for three difficulty levels for each task
    specified in the `hue_order` list. The four predefined tasks are:
    ['face matching', 'delay', 'math', 'interference']. Each task is associated
    with a specific colour, and the three difficulty levels are represented by
    varying shades of that colour using different alpha values.

    Parameters
    ----------
    hue_order: list of str, list of task names for which the palette is to be created. The input
        task names may include the predefined tasks:
        ['face matching', 'delay', 'math', 'interference'], or their alternative names:
        ['Matching', 'Unfilled Delay', 'Filled Delay - emotions', 'Filled Delay - math'].

    Returns
    -------
    flattened_palette: list of tuple, list of RGBA tuples representing the colours for each task at three
        difficulty levels. Each task has three corresponding colours in the order of difficulty levels (high, medium, low).

    Notes
    -----
    - The predefined task names are mapped to their corresponding colours as follows:
        - 'face matching': Dark blue shades.
        - 'delay': Orange shades.
        - 'math': Teal shades.
        - 'interference': Red shades.
    - The difficulty levels are represented by varying alpha values: [1, 0.65, 0.33].
    - If a task in `hue_order` does not match any key in the reverse mapping
      dictionary, it will retain its original name.
    """
    # have a reverse mapping dic done
    rename_tasks_dic_reverse = {'Matching': 'face matching',
                                'Unfilled Delay': 'delay',
                                'Filled Delay - emotions': 'interference',
                                'Filled Delay - math': 'math'}

    # define a dictionary to map each task to its corresponding colour
    color_map = {
        'face matching': [(0, 0, 0.55)] * 3,
        'delay': [(0.94, 0.57, 0.095)] * 3,
        'math': [(0.0, 0.5, 0.5)] * 3,
        'interference': [(0.79, 0.093, 0.11)] * 3,
    }

    # define corresponding alphas to have different shades for difficulty
    alphas = [1, 0.65, 0.33]

    # replace the hue_order elements with mapped labels
    hue_order_mapped = [rename_tasks_dic_reverse.get(hue, hue) for hue in hue_order]

    # select the corresponding colors based on hue_order
    selected_palette = [color_map[hue] for hue in hue_order_mapped]

    #     # select the corresponding colours based on hue_order
    #     selected_palette = [color_map[hue] for hue in hue_order]

    # flatten the palette with the corresponding alphas
    flattened_palette = [
        (r, g, b, alpha)
        for colors in selected_palette
        for (r, g, b), alpha in zip(colors, alphas)
    ]

    return flattened_palette


def plot_significance_pairs(num1, num2, p, ax,
                            centers=None, heights=None, yerr=None, dh=0.05, barh=0.05, x_offset=0, fs=None):
    """
    Annotate a bar plot with brackets and p-values.
    Inspired by https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph

    This function adds brackets and significance annotations (e.g., p-values)
    to a bar plot, indicating statistical comparisons between two bars.

    Parameters
    ----------
    num1: int, Index of the left bar to put the bracket over.
    num2: int, Index of the right bar to put the bracket over.
    p: float or str, p-value for the comparison, or a string to write directly. If a float is provided, it will generate asterisks based on significance.
    centers: list of float, optional, centers of all bars (x-coordinates, like plt.bar() input). If not provided, will be extracted from the plot.
    heights: list of float, optional, heights of all bars (like plt.bar() input). If not provided, will be extracted from the plot.
    yerr: list of float or bool, optional, error values for all bars. If set to True, will attempt to extract from
        the plot. If False or None, no error bars are considered.
    dh: float, optional, height offset over the bars (in axis coordinates, from 0 to 1). Default is 0.05.
    barh: float, optional, height of the bracket (in axis coordinates, from 0 to 1). Default is 0.05.
    x_offset: float, optional, default to 0, how far from the center the line starts.
    fs: int, optional, font size for the significance annotation text. If None, default font size is used.


    """

    # Determine the marker for the p-value
    if isinstance(p, str):
        p_value_marker = p
    else:
        if p >= 0.05:
            p_value_marker = 'n.s.'
        elif p < 0.001:
            p_value_marker = "***"
        elif p < 0.01:
            p_value_marker = "**"
        else:
            p_value_marker = "*"

    # Extract bar patches from the plot
    bars = ax.patches

    # Extract centers and heights if not provided
    if centers is None:
        centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    if heights is None:
        heights = [bar.get_height() for bar in bars]

    # Determine positions for the left and right bars
    lx, ly = centers[num1], heights[num1]
    rx, ry = centers[num2], heights[num2]

    # Swap if necessary to ensure lx < rx, otherwise the offset won't work
    if lx > rx:
        lx, rx = rx, lx

    # update the center if there is an offset
    lx += x_offset
    rx -= x_offset

    # Adjust heights for error bars, if any
    if yerr:
        if isinstance(yerr, bool):
            # Extract error bar heights from the plot if 'yerr' is True
            yerr = [patch.get_y() + patch.get_height() for patch in ax.patches]

        ly += yerr[num1]
        ry += yerr[num2]

    # Convert offsets to data coordinates
    ax_y0, ax_y1 = ax.get_ylim()  # plt.gca().get_ylim()

    # multiply by the range of y-values in the plot to get a meaningful offset in the data's scale.
    dh *= (ax_y1 - ax_y0)  # 'dh' is now the offset height above the bar in data units
    barh *= (ax_y1 - ax_y0)  # 'barh' is now the height of the bracket in data units

    # Determine y-coordinate for the top of the bracket - find the highest point between the two bars being compared,
    # including any error bars, and add the calculated offset 'dh' to place the bracket above the taller bar.
    y = max(ly, ry) + dh

    # Define coordinates for the bracket
    # 'barx' defines the x-coordinates of the bracket's corners: it starts at the left bar's x (lx),
    # goes up vertically to the bracket height, then horizontally to the right bar's x (rx),
    # and finally down again to the y-coordinate where the bracket ends.
    barx = [lx, lx, rx, rx]

    # 'bary' defines the y-coordinates of the bracket's corners: starting from the calculated 'y'
    # position, goes up by 'barh' to reach the top of the bracket, stays at that height between
    # the bars, and then goes down again to 'y'.
    bary = [y, y + barh, y + barh, y]

    # 'mid' calculates the midpoint for placing the text annotation; it's the horizontal midpoint
    # between the two bars (average of 'lx' and 'rx'), and vertically just above the bracket's top.
    mid = ((lx + rx) / 2, y + barh)

    # Plot the bracket
    ax.plot(barx, bary, c='black', linewidth=1)

    # Set text properties for the p-value annotation
    kwargs = {'ha': 'center', 'va': 'bottom'}
    if fs is not None:
        kwargs['fontsize'] = fs

    # Add the p-value annotation, mid is a tuple containing the x and y coordinates where the text annotation
    # (p-value or significance marker) will be placed on the plot. * will unpack the tuple
    ax.text(*mid, p_value_marker, **kwargs)

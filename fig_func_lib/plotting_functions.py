from __future__ import division
import numpy as np
import ms_func_lib.Kd_fitting_functions as cff
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import operator
from matplotlib import colors
from scipy import stats
import networkx as nx

def frmt(value):
    if type(value) == str or value is None:
        return value

    if value > 1000 or value < 0.0001:
        return str.format('{0:.0e}', value)
    elif value >= 1000 and value > 100:
        return str.format('{:.0f}', value)
    elif value >= 100 and value >= 10:
        return str.format('{:.1f}', value)
    elif value >= 100 and value >= 10:
        return str.format('{:.2f}', value)
    else:
        return str.format('{:.3f}', value)


def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height], axisbg=axisbg)

    return subax


def convert_ticks_to_log_format(ax,type='both'):
    if type == 'x' or type == 'both':
        xtick_vals = ax.get_xticks().tolist()
        xtick_vals = np.asarray(xtick_vals).astype(int)
        new_xtick_labels = [str(10**np.float32(x)) if x<1 else str(int(10**np.float32(x))) for x in xtick_vals]
        ax.set_xticklabels(new_xtick_labels)
    if type == 'y' or type == 'both':
        ytick_vals = ax.get_yticks().tolist()
        ytick_vals = np.asarray(ytick_vals).astype(int)
        new_ytick_labels = ['10$^{'+str(x)+'}$' for x in ytick_vals]
        ax.set_yticklabels(new_ytick_labels)


def convert_log_ticks_to_lin_format(ax,type='both'):
    if type == 'x' or type == 'both':
        xtick_vals = ax.get_xticks().tolist()
        xtick_vals = 10**np.asarray(xtick_vals)
        new_xtick_labels = [str(x) for x in xtick_vals]
        ax.set_xticklabels(new_xtick_labels)
    if type == 'y' or type == 'both':
        ytick_vals = ax.get_yticks().tolist()
        ytick_vals = 10**np.asarray(ytick_vals)
        new_ytick_labels = [str(x) for x in ytick_vals]
        ax.set_yticklabels(new_ytick_labels)


def set_plot_location(gridsize_y, gridsize_x, gridpos_y, gridpos_x, rowspan=1, colspan=1):
    # plot location: dict w/ keys: gridsize_x, gridsize_y, gridpos_x, gridpos_y, rowspan, colspan
    # use set_plot_location() to create input for subplot2grid
    # subplot2grid(shape, loc, rowspan=1, colspan=1)
    # subplot2grid: Shape of grid in which to place axis. First entry is number of rows, second entry is number of columns.
    # then second list: Location to place axis within grid. First entry is row number, second entry is column number.
    # subplot2grid([numrows,numcols],[rowcoord,colcoord], rowspan, colspan)
    plot_location = dict()
    plot_location['gridsize_x'] = gridsize_x
    plot_location['gridsize_y'] = gridsize_y
    plot_location['gridpos_x'] = gridpos_x
    plot_location['gridpos_y'] = gridpos_y
    plot_location['rowspan'] = rowspan
    plot_location['colspan'] = colspan
    return plot_location


def set_fit_parameters(row):
    from collections import defaultdict
    # fit parameters set from iterating over rows in df

    fit_params = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    fit_params['linear']['slope'] = row.linear_slope
    fit_params['linear']['offset'] = row.linear_offset

    fit_params['fo']['Fmax'] = row.fo_Fmax
    fit_params['fo']['Kd'] = row.fo_Kd
    fit_params['fo']['offset'] = row.fo_offset

    fit_params['best_fit']['fit_method'] = row.fit_method
    fit_params['best_fit']['fit_Kd'] = row.fit_Kd
    fit_params['best_fit']['fit_offset'] = row.fit_offset

    if row.fit_method == 'linear':
        fit_params['best_fit']['fit_slope'] = row.fit_slope
    else:
        fit_params['best_fit']['fit_slope'] = None

    fit_params['best_fit']['fit_Fmax'] = row.fit_Fmax
    fit_params['best_fit']['fit_rsq'] = row.fit_rsq
    fit_params['best_fit']['fit_AICc'] = row.fit_AICc

    return fit_params


def get_fig_data(row):
    fig_data = dict()

    fig_data['peptide'] = row.gene_name
    fig_data['domain'] = row.domain
    fig_data['pY_pos'] = str(row.pY_pos)
    fluorescence = np.asarray(row.FluorTxt.split(',')).astype('float')
    pepconc = np.asarray(row.pepConcTxt.split(',')).astype('float')
    fluorescence = fluorescence[~np.isnan(pepconc)]
    pepconc = pepconc[~np.isnan(pepconc)]

    fig_data['title'] = 'Kd Fits'
    fig_data['x_data'] = pepconc
    fig_data['y_data'] = fluorescence
    fig_data['bg'] = row.bg
    fig_data['xlabel'] = '$[domain]$'
    fig_data['ylabel'] = '$Fluorescence$'
    fig_data['filename'] = row.filename
    fig_data['domain_pos'] = row.domain_pos
    fig_data['plate_number'] = row.plate_number
    fig_data['fo_Fmax'] = row.fo_Fmax + row.fo_offset
    fig_data['fo_fitKd'] = row.fo_Kd
    fig_data['fo_AICc'] = row.fo_AICc
    fig_data['linear_AICc'] = row.linear_AICc
    fig_data['fo_offset'] = row.fo_offset
    return fig_data


def generate_yfits(fig_data, fit_params, steps=1000):
    y_fits = dict()

    # y_fits['x_fit'] = np.append(np.linspace(0.001, 0.01, steps // 2),
    #                             np.linspace(0, max(fig_data['x_data']) * 1.1, steps))
    y_fits['x_fit'] = np.append(np.linspace(0.001, 0.1, steps),
                                 np.linspace(0.2, 10, steps))
    y_fits['x_fit'] = np.append(y_fits['x_fit'],np.linspace(11, 1100, steps))


    y_fits['fo'] = cff.model_firstorder_offset(y_fits['x_fit'], fit_params['fo']['Fmax'], fit_params['fo']['Kd'],
                                               fit_params['fo']['offset'])
    y_fits['fo_fig'] = cff.model_firstorder_offset(fig_data['x_data'], fit_params['fo']['Fmax'], fit_params['fo']['Kd'],
                                                   fit_params['fo']['offset'])
    y_fits['fo_label'] = 'fo'
    y_fits['fit_Fmax'] = fit_params['best_fit']['fit_Fmax']
    y_fits['fit_Kd'] = fit_params['best_fit']['fit_Kd']
    y_fits['fit_offset'] = fit_params['best_fit']['fit_offset']

    y_fits['linear'] = (fit_params['linear']['slope'] * y_fits['x_fit']) + fit_params['linear']['offset']
    y_fits['linear_fig'] = (fit_params['linear']['slope'] * fig_data['x_data']) + fit_params['linear']['offset']
    y_fits['linear_label'] = 'linear'

    y_fits['linear_residuals'] = y_fits['linear_fig'] - fig_data['y_data']
    y_fits['fo_residuals'] = y_fits['fo_fig'] - fig_data['y_data']

    return y_fits


def plot_residuals(fig_data, y_fits, plot_location, logplots=False):
    # fig_data: dict with keys: x_data, y_data, title
    # plot location: dict w/ keys: gridsize_x, gridsize_y, gridpos_x, gridpos_y, rowspan
    # use set_plot_location() to create
    ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                          (plot_location['gridpos_y'], plot_location['gridpos_x']), rowspan=plot_location['rowspan'],
                          colspan=plot_location['colspan'])
    title_text = 'Residual Plots'

    if logplots == True:
        xvalues = np.log2(fig_data['x_data'])
        title_text = title_text + ' (log)'
    else:
        xvalues = fig_data['x_data']

    ax.set_title(title_text, size=6)

    ind = xvalues  # the x locations for the groups
    width = 0.2  # the width of the bars

    rects0 = ax.bar(ind - width, y_fits['linear_residuals'], width / 2, color='b', edgecolor='none')
    rects1 = ax.bar(ind - (width/2), y_fits['fo_residuals'], width / 2, color='r', edgecolor='none')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Residuals', size=8)
    ax.set_xticks(ind)

    ax.legend((rects0[0], rects1[0]), ('linear', 'fo'), loc='upper right', fontsize=5)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
        tick.label.set_rotation('vertical')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)


def generate_tbldata_fitresults(row):
    # 0 Kd
    # 1 Fmax
    # 2 offset
    # 3 slope
    # 4 AICc
    # 5 rsq
    # 6 dw
    # 7 resVar
    # 8 success
    linear_data = [None,
                   None,
                   row.linear_offset,
                   row.linear_slope,
                   row.linear_AICc,
                   row.linear_rsq,
                   row.linear_resVar,
                   row.linear_success,
                   ]

    fo_data = [row.fo_Kd,
               row.fo_Fmax,
               row.fo_offset,
               None,
               row.fo_AICc,
               row.fo_rsq,
               row.fo_resVar,
               row.fo_success
               ]

    grouped_data_list = [linear_data, fo_data]
    table_data = []
    for item in grouped_data_list:
        table_data.append([
            frmt(item[0]),
            frmt(item[1]),
            frmt(item[2]),
            frmt(item[3]),
            frmt(item[4]),
            frmt(item[5]),
            frmt(item[6]),
            item[7],
        ])
    table_col_labels = ['$K_d$', '$F_{max}$', 'offset', 'slope', 'AICc', '$r^2$', 'resVar', 'success']
    table_row_labels = ['linear', 'fo']
    table_col_widths = [0.08] * 8

    tbl_specs = dict()
    tbl_specs['data'] = table_data
    tbl_specs['col_labels'] = table_col_labels
    tbl_specs['row_labels'] = table_row_labels
    tbl_specs['col_widths'] = table_col_widths

    return tbl_specs


def generate_tbldata_chosenfit(row):
    table_data = [
        [row.fit_method,
         row.binding_call,
         str(frmt(row.fit_Kd)),
         str(frmt(row.fit_Fmax)),
         str(frmt(row.fit_offset)),
         str(frmt(row.fit_slope)),
         str(frmt(row.fit_AICc)),
         str(frmt(row.fit_rsq)),
         str(frmt(row.fit_resVar)),
         str(frmt(row.fit_modelDNR)),
         str(frmt(row.fit_sumres)),
         str(frmt(row.fit_snr)),
         str(row.fit_success)],
    ]
    table_col_labels = ['Best Fit', 'Binding Call', 'K$_d$', 'F$_{max}$', 'offset', 'slope', 'AICc', '$r^2$',
                        'resVar', 'modelDNR', 'sumres', 'snr', 'Success']
    table_col_widths = [0.07] + [0.10] + [0.08] * 11

    tbl_specs = dict()
    tbl_specs['data'] = table_data
    tbl_specs['col_labels'] = table_col_labels
    tbl_specs['col_widths'] = table_col_widths
    tbl_specs['row_labels'] = None

    return tbl_specs


def generate_tbldata_id(row):
    table_data = [[
        str(row.domain),
        str(row.gene_name),
        str(row.pY_pos),
        str(row.filename),
        str(row.plate_number),
        str(row.domain_pos),
    ]]
    table_col_labels = ['domain', 'peptide source', 'pY pos', 'filename', 'plate num', 'domain pos']
    table_col_widths = [0.10, 0.10, 0.10, 0.50, 0.10, 0.10]

    tbl_specs = dict()
    tbl_specs['data'] = table_data
    tbl_specs['col_labels'] = table_col_labels
    tbl_specs['col_widths'] = table_col_widths
    tbl_specs['row_labels'] = None

    return tbl_specs


def generate_tbldata_replicates(df):
    text_cols = [
        'filename',
        'plate_number',
        'domain_pos',
        'binding_call',
    ]

    data_cols = [
        'fit_method',
        'fit_Kd',
        'fit_Fmax',
        'fit_offset',
        'fit_slope',
        'fit_rsq',
        'fit_resVar',
        'fit_snr',
        'fit_modelDNR',
    ]

    table_df = df[text_cols + data_cols].copy()
    table_df[data_cols] = table_df[data_cols].applymap(frmt)

    table_col_labels = [
        'file',
        'plt',
        'dpos',
        'bind call',
        'fit',
        'Kd',
        'Fmax',
        'c',
        'slope',
        'rsq',
        'resVar',
        'snr',
        'modDNR',
    ]

    table_col_widths = [0.40, 0.04, 0.04, 0.08, 0.05, 0.06, 0.07, 0.05, 0.05, 0.05, 0.07, 0.05, 0.08]

    tbl_specs = dict()
    tbl_specs['data'] = table_df.values
    tbl_specs['col_labels'] = table_col_labels
    tbl_specs['col_widths'] = table_col_widths
    tbl_specs['row_labels'] = None
    return tbl_specs


def plot_table(df, plot_location, table_type):
    # ID table
    ax = plt.subplot2grid((plot_location['gridsize_x'], plot_location['gridsize_y']),
                          (plot_location['gridpos_x'], plot_location['gridpos_y']), rowspan=plot_location['rowspan'],
                          colspan=plot_location['colspan'])
    ax.set_axis_off()
    if table_type == 'replicates':
        tbl_data = generate_tbldata_replicates(df)
    if table_type == 'id':
        tbl_data = generate_tbldata_id(df)
    if table_type == 'chosenfit':
        tbl_data = generate_tbldata_chosenfit(df)
    if table_type == 'fulldata':
        tbl_data = generate_tbldata_fitresults(df)

    the_table = ax.table(cellText=tbl_data['data'], colLabels=tbl_data['col_labels'], colWidths=tbl_data['col_widths'],
                         rowLabels=tbl_data['row_labels'], loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(6)


def get_index_of_filtered_replicates(df, group_cols, filter_specs, filter_type, min_count_per_replicate_group=1):
    def createComparator(comparison_operator_text, input2):
        if comparison_operator_text == 'count':
            return lambda x: x.count() >= input2
        op_dict = {'<': operator.lt,
                   '<=': operator.le,
                   '==': operator.eq,
                   '!=': operator.ne,
                   '>=': operator.ge,
                   '>': operator.gt
                   }
        comparator_op = op_dict[comparison_operator_text]
        if filter_type == 'all':
            return lambda x: all(comparator_op(x, input2))
        elif filter_type == 'any':
            return lambda x: any(comparator_op(x, input2))
        else:
            raise ValueError('Improper filter type: filter type should be "all" or "any"')

    data_cols = []
    filter_column_names = []
    filter_dict = defaultdict(dict)

    # filter for replicate groups with more than one measurement in them
    filter_dict['fit_Kd']['multiple_replicates'] = createComparator('count', min_count_per_replicate_group)

    filter_column_names.append('multiple_replicates')
    data_cols.append('fit_Kd')

    for data_field, operator_text, filter_criterion in filter_specs:
        filt_col_name = str(data_field) + operator_text + str(filter_criterion)
        filter_column_names.append(filt_col_name)
        if data_field not in data_cols:
            data_cols.append(data_field)
        filter_dict[data_field][filt_col_name] = createComparator(operator_text, filter_criterion)
        filter_dict[data_field][data_field + '_values'] = lambda x: tuple(x)
    replicates_df = df[group_cols + data_cols].groupby(group_cols).agg(filter_dict)
    replicates_df.columns = replicates_df.columns.droplevel(0)
    return replicates_df[replicates_df[filter_column_names].all(axis=1)].index


def theoretical_Kd_plot(plot_location, Kd=1, Fmax=100, offset=0, logplot=False):
    ## One to One Plot, Theoretical 1uM
    ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                          (plot_location['gridpos_y'], plot_location['gridpos_x']), rowspan=plot_location['rowspan'],
                          colspan=plot_location['colspan'])
    conc = np.append(np.linspace(0.001, 0.1, 1000),
                     np.linspace(0.2, 10, 1000))
    conc = np.append(conc, np.linspace(11, 1100, 1000))

    y_vals = cff.model_firstorder_offset(conc, Fmax + offset, Kd, offset)

    if logplot == True:
        ax.set_title('Semi-Log Plot (x-axis, $log_{10}$)',size=8)
        ax.plot(np.log10(conc), y_vals, color='k')
        ax.set_xlim([-3.2, 3.2])
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        convert_ticks_to_log_format(ax, 'x')
        ax.set_ylim([0.9 * offset, 1.1 * (offset + Fmax)])
        ax.set_yticks([Fmax / 2, Fmax])
        ax.set_yticklabels(['$\\frac{1}{2}$$F$$_{max}$', '$F$$_{max}$'], size=8)
        ax.set_xlabel('$log_{10}$ $\mu$M',size=7)
        ax.axhline(Fmax, color='g', linestyle='--', linewidth=1)
        ax.plot([0, 0], [0, Fmax / 2], color='dodgerblue', linestyle='--', linewidth=1)
        ax.plot([-3, 0], [Fmax / 2, Fmax / 2], color='dodgerblue', linestyle='--', linewidth=1)
        plt.text(np.log10(Kd)+0.1, 20, '$K_d$ =  1$\mu$$M$', color='k', size=8)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)

    else:
        ax.set_title('Linear Plot',size=8)
        ax.plot(conc, y_vals, color='k')
        ax.set_xlim([-0.1, 10.1])
        ax.set_ylim([0.9 * offset, 1.1 * (offset + Fmax)])
        ax.set_yticks([Fmax / 2, Fmax])
        ax.set_yticklabels(['$\\frac{1}{2}$$F$$_{max}$', '$F$$_{max}$'], size=8)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xticklabels(['0', '2', '4', '6', '8', '10'], size=6)
        ax.set_xlabel('$\mu$M',size=7)
        ax.axhline(Fmax, color='g', linestyle='--', linewidth=1)
        ax.plot([1, 1], [0, Fmax / 2], color='dodgerblue', linestyle='--', linewidth=1)
        ax.plot([0, 1], [Fmax / 2, Fmax / 2], color='dodgerblue', linestyle='--', linewidth=1)
        plt.text(1.2, 20, '$K_d$ =  1$\mu$$M$', color='k', size=8)

        # ax.set_xlabel('Protein Concentration ($\mu$$M$)',size=8)

    return ax


def plot_Kd_curves_v2(fits_to_plot, fig_data, y_fits, plot_location, fig_title, logplots=False, ylim='on', legend='on', tick_disp='on', bg_disp = 'off'):
    # fits to plot is a list of strings, e.g. ['linear', 'fo']
    # fig_data: dict with keys: x_data, y_data, title
    # plot location: dict w/ keys: gridsize_x, gridsize_y, gridpos_x, gridpos_y, rowspan
    # use set_plot_location() to create
    # subplot2grid: Shape of grid in which to place axis. First entry is number of rows, second entry is number of columns.
    # then second tuple: Location to place axis within grid. First entry is row number, second entry is column number.
    ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                          (plot_location['gridpos_y'], plot_location['gridpos_x']), rowspan=plot_location['rowspan'],
                          colspan=plot_location['colspan'])
    if logplots == False:
        ax.scatter(fig_data['x_data'], fig_data['y_data'], marker='o', s=20, facecolor='none', edgecolor='k',
                   linewidths=0.5)
        if max(fig_data['x_data']) >= 10:
            ax.set_xlim([-0.5, 10.5])
            ax.set_xticks([0, 2, 4, 6, 8, 10])
        else:
            ax.set_xlim([-0.5, 5.5])
            ax.set_xticks([0,1,2,3,4,5])
    elif logplots == True:
        ax.scatter(np.log10(fig_data['x_data']), fig_data['y_data'], marker='o', s=20, facecolor='none', edgecolor='k',
                   linewidths=0.5)
        #ax.set_xlim([-3.2, 1.2])
        ax.set_xticks([-3,-2,-1,0,1])
        convert_ticks_to_log_format(ax, 'x')

    title_text = fig_title  # fig_data['title']
    if logplots == True:
        title_text = title_text + ' (log10)'
    ax.set_title(title_text, size=6)

    fit_style = ['--b', '-.r', '--g', '-.y', '-.b', '-.r']

    if logplots == True:
        xvalues = np.log10(y_fits['x_fit'])
    else:
        xvalues = y_fits['x_fit']
    for i, fit_type in enumerate(fits_to_plot):
        if fit_type == 'linear':
            ax.plot(xvalues, y_fits['linear'], fit_style[i], label=y_fits['linear_label'], linewidth=0.5)
        if fit_type == 'fo':
            ax.plot(xvalues, y_fits['fo'], fit_style[i], label=y_fits['fo_label'], linewidth=0.5)

    if ylim == 'on':
        ax.set_ylim([100, 400])
    if ylim == 'off':
        ylim_minvalue = np.min([fig_data['bg'], np.min(fig_data['y_data'])]) * 0.98
        ylim_maxvalue = np.max([fig_data['bg'], np.max(fig_data['y_data'])]) * 1.02
        ax.set_ylim([ylim_minvalue, ylim_maxvalue])

    #ax.axhline(y_fits['fit_offset'], linestyle='--', color='gray', linewidth = 0.5)
    ax.plot(xvalues, [y_fits['fit_offset']]*len(xvalues), linestyle=':', color='gray', label='fit offset', linewidth=0.5)
    if bg_disp == 'on':
        ax.plot(xvalues, [fig_data['bg']]*len(xvalues), linestyle=':', color='orange', label='pub bg', linewidth=0.5)
        #ax.axhline(fig_data['bg'], linestyle=':', color='k', linewidth = 0.5)
    if legend == 'on':
        ax.legend(loc='upper left', fontsize=5)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
        tick.label.set_rotation('vertical')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    if tick_disp == 'off':
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticks([])


def plot_Kd_curves_diagnostic(fig_data, y_fits, plot_location, fig_title, logplots=False, ylim='on', legend='on', tick_disp='on', bg_disp = 'off',Kd_line='off',Fmax_line='off',xlabel_disp='off',ylabel_disp='off'):
    # fig_data: dict with keys: x_data, y_data, title
    # plot location: dict w/ keys: gridsize_x, gridsize_y, gridpos_x, gridpos_y, rowspan
    # use set_plot_location() to create
    # subplot2grid: Shape of grid in which to place axis. First entry is number of rows, second entry is number of columns.
    # then second tuple: Location to place axis within grid. First entry is row number, second entry is column number.
    ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                          (plot_location['gridpos_y'], plot_location['gridpos_x']), rowspan=plot_location['rowspan'],
                          colspan=plot_location['colspan'])
    title_text = fig_title
    ylabel_text = 'FP (mP)'
    if logplots == True:
        ax.scatter(np.log10(fig_data['x_data']), fig_data['y_data'], marker='o', s=20, facecolor='none', edgecolor='k',
                   linewidths=0.5)
        # ax.set_xlim([-3.2, 1.2])
        ax.set_xticks([-3, -2, -1, 0, 1, 2])
        convert_ticks_to_log_format(ax, 'x')
        #title_text = title_text + ' SemiLog'
        xlabel_text = 'Concentration ($\mu$M, $log_{10}$) '
        xvalues = np.log10(y_fits['x_fit'])
    else:
        ax.scatter(fig_data['x_data'], fig_data['y_data'], marker='o', s=20, facecolor='none', edgecolor='k',
                   linewidths=0.5)
        if max(fig_data['x_data']) >= 5:
            ax.set_xlim([-0.5, 10.5])
            ax.set_xticks([0, 2, 4, 6, 8, 10])
        else:
            ax.set_xlim([-0.5, 5.5])
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            #    title_text = title_text + ' Linear'
        xlabel_text = 'Concentration ($\mu$M)'
        xvalues = y_fits['x_fit']

    if xlabel_disp == 'on':
        ax.set_xlabel(xlabel_text,size=6)
    else:
        pass

    if ylabel_disp == 'on':
        ax.set_ylabel(ylabel_text,size=6)
    else:
        pass

    ax.plot(xvalues, y_fits['fo'], '--b', label=y_fits['fo_label'], linewidth=0.5)

    if ylim == 'on':
        ylim_minvalue = 100
        ylim_maxvalue = 400
        ax.set_ylim([ylim_minvalue, ylim_maxvalue])
    else: # if ylim == 'off':
        ylim_minvalue = np.min([fig_data['bg'], np.min(fig_data['y_data'])]) * 0.98
        ylim_maxvalue = np.max([y_fits['fit_Fmax']+y_fits['fit_offset'], np.max(fig_data['y_data'])]) * 1.02
        ax.set_ylim([ylim_minvalue, ylim_maxvalue])

    ax.axhline(y_fits['fit_offset'], linestyle='--', color='gray', label='fit offset', linewidth = 0.5)
    if Kd_line == 'on':
        half_Fmax =  y_fits['fit_offset'] + (y_fits['fit_Fmax'] / 2)
        if logplots == True:
            x_loc = np.log10(y_fits['fit_Kd'])
            ax.plot([x_loc, x_loc], [0, half_Fmax], color='firebrick', linestyle='--', linewidth=0.5)
            offset = 0.3
        else:
            x_loc = y_fits['fit_Kd']
            ax.plot([x_loc, x_loc], [0, half_Fmax], color='firebrick', linestyle='--', linewidth=0.5)
            offset = 0.5
        ax.text(x_loc+offset,half_Fmax,'Kd = '+str(round(y_fits['fit_Kd'],2)),size=6,color='firebrick')
    if Fmax_line == 'on':
        Fmax_y = y_fits['fit_offset'] + y_fits['fit_Fmax']
        ax.axhline(Fmax_y, linestyle='--', color='g', label='fit Fmax', linewidth=0.5)
    if bg_disp == 'on':
        ax.axhline(fig_data['bg'], linestyle=':', color='orange', linewidth = 0.5, label='pub bg')
    if legend == 'on':
        ax.legend(loc='lower right', fontsize=5)



    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
        tick.label.set_rotation('vertical')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    if tick_disp == 'off':
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticks([])


def plot_Kd_curves_examples(fits_to_plot, fig_data, y_fits, plot_location, logplots=False, ylim='on', legend='on', tick_disp='on', offset = 'off', bg_disp = 'off', fmax='off', kd='off', legendloc='upper left'):
    # fig_data: dict with keys: x_data, y_data, title
    # plot location: dict w/ keys: gridsize_x, gridsize_y, gridpos_x, gridpos_y, rowspan, colspan
    # use set_plot_location() to create
    # subplot2grid: Shape of grid in which to place axis. First entry is number of rows, second entry is number of columns.
    # then second tuple: Location to place axis within grid. First entry is row number, second entry is column number.
    ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                          (plot_location['gridpos_y'], plot_location['gridpos_x']), rowspan=plot_location['rowspan'],
                          colspan=plot_location['colspan'])
    if logplots == True:
        ax.scatter(np.log10(fig_data['x_data']), fig_data['y_data'], marker='o', s=20, facecolor='none', edgecolor='k',
                   linewidths=0.5)
        xvalues = np.log10(y_fits['x_fit'])
        xlim_maxvalue = 3.2
        ax.set_xlim([-3.2, xlim_maxvalue])
        ax.set_xticks([-3,-2,-1,0,1,2,3])
        convert_ticks_to_log_format(ax, 'x')
    else:
        ax.scatter(fig_data['x_data'], fig_data['y_data'], marker='o', s=20, facecolor='none', edgecolor='k',
                   linewidths=0.5)
        xvalues = y_fits['x_fit']
        if max(fig_data['x_data']) >= 10:
            xlim_maxvalue = 10.5
            ax.set_xlim([-0.5, xlim_maxvalue])
            ax.set_xticks([0, 2, 4, 6, 8, 10])
        else:
            xlim_maxvalue = 5.5
            ax.set_xlim([-0.5, xlim_maxvalue])
            ax.set_xticks([0,1,2,3,4,5])

    if bg_disp == 'on':
        ylim_minvalue = np.min([np.min(fig_data['y_data']),np.min(fig_data['bg'])]) * 0.95
    else:
        ylim_minvalue = np.min([np.min(fig_data['y_data'])]) * 0.95
    ylim_maxvalue = np.max([fig_data['bg'], np.max(fig_data['y_data']), fig_data['fo_Fmax']]) * 1.1

    if ylim == 'off':
        ax.set_ylim([ylim_minvalue*.9, ylim_maxvalue])

    if fmax == 'on':
        ax.axhline(fig_data['fo_Fmax'],linestyle = '--', color='g', linewidth=0.5)
        if logplots == True:
            ax.text(xlim_maxvalue*1.05, fig_data['fo_Fmax']-5, '$F_{max}$', size=8, color='g')
            half_fmax = fig_data['fo_offset']+(fig_data['fo_Fmax']-fig_data['fo_offset'])/2
            ax.axhline(half_fmax, linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, half_fmax, '$\\frac{1}{2}$$F$$_{max}$', size=8, color='g')
            ax.axhline(fig_data['fo_offset'], linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, fig_data['fo_offset'], '$F_0$', size=8, color='g')
        else:
            ax.text(xlim_maxvalue*1.01,fig_data['fo_Fmax']-5,'$F_{max}$',size=8,color='g')
            half_fmax = fig_data['fo_offset']+(fig_data['fo_Fmax']-fig_data['fo_offset'])/2
            ax.axhline(half_fmax, linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, half_fmax, '$\\frac{1}{2}$$F$$_{max}$', size=8, color='g')
            ax.axhline(fig_data['fo_offset'], linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, fig_data['fo_offset'], '$F_0$', size=8, color='g')

    if kd == 'on':
        if logplots == True:
            ax.text(np.log10(fig_data['fo_fitKd']), ylim_maxvalue*1.01, '$K_d$', size=8, color='g')
            ax.axvline(np.log10(fig_data['fo_fitKd']), linestyle='--', color='g', linewidth=0.5)
        else:
            if fig_data['fo_fitKd'] < xlim_maxvalue:
                ax.text(fig_data['fo_fitKd'],ylim_maxvalue*1.01,'$K_d$',size=8,color='g')
                ax.axvline(fig_data['fo_fitKd'], linestyle='--', color='g', linewidth=0.5)

    if ylim == 'on':
        ax.set_ylim([100, 400])


    if offset == 'on':
        ax.plot(xvalues, [y_fits['fit_offset']]*len(xvalues), linestyle=':', color='gray', label='fit offset', linewidth=1)

    if bg_disp == 'on':
        ax.plot(xvalues, [fig_data['bg']]*len(xvalues), linestyle=':', color='orange', label='pub bg', linewidth=1)

    if legend == 'on':
        ax.legend(loc=legendloc, fontsize=5, frameon=False)

    fit_style = ['-b', '-r']

    for i, fit_type in enumerate(fits_to_plot):
        if fit_type == 'linear':
            ax.plot(xvalues, y_fits['linear'], fit_style[i], label=y_fits['linear_label'], linewidth=0.5)
        if fit_type == 'fo':
            ax.plot(xvalues, y_fits['fo'], fit_style[i], label=y_fits['fo_label'], linewidth=0.5)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
        #tick.label.set_rotation('vertical')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    if tick_disp == 'off':
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticks([])


def plot_Kd_curves_examples_v2(fits_to_plot, fig_data, y_fits, plot_location, logplots=False, type=None, title=None, ylabel=None, xlabel=False, Kd_label=False,Kd_fixedlabel=False):
    # fig_data: dict with keys: x_data, y_data, title
    # plot location: dict w/ keys: gridsize_x, gridsize_y, gridpos_x, gridpos_y, rowspan, colspan
    # use set_plot_location() to create
    # subplot2grid: Shape of grid in which to place axis. First entry is number of rows, second entry is number of columns.
    # then second tuple: Location to place axis within grid. First entry is row number, second entry is column number.

    if type == 'SNR':
        ylim = 'off'
        offset = 'off'
        bg_disp = 'off'
        fmax = 'on'
        kd = 'off'
        legend = 'off'
        legendloc = 'upper left'
        tick_disp = 'on'
        fit_line_style = '--'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5

    elif type == 'model_selection':
        ylim = 'on'
        offset = 'off'
        bg_disp = 'off'
        fmax = 'off'
        kd = 'off'
        legend = 'on'
        legendloc = 'upper left'
        tick_disp = 'on'
        fit_line_style = '--'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5


    elif type == 'binding':
        ylim = 'off'
        offset = 'off'
        bg_disp = 'off'
        fmax = 'off'
        kd = 'off'
        legend = 'on'
        legendloc = 'upper left'
        tick_disp = 'on'
        fit_line_style = '--'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5

    elif type == 'binding_zoomed':
        ylim = 'zoom'
        offset = 'off'
        bg_disp = 'off'
        fmax = 'off'
        kd = 'off'
        legend = 'on'
        legendloc = 'lower right'
        tick_disp = 'on'
        fit_line_style = '--'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5

    elif type == 'binding_ylimon':
        ylim = 'on'
        offset = 'off'
        bg_disp = 'off'
        fmax = 'off'
        kd = 'off'
        legend = 'on'
        legendloc = 'upper left'
        tick_disp = 'on'
        fit_line_style = '--'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5


    elif type == 'bg':
        ylim = 'off'
        offset = 'on'
        bg_disp = 'on'
        fmax = 'off'
        kd = 'off'
        legend = 'off'
        #legendloc = 'upper left'
        tick_disp = 'on'
        fit_line_style = '--'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5


    elif type == 'Kd':
        ylim = 'off'
        offset = 'off'
        bg_disp = 'off'
        fmax = 'full'
        kd = 'on'
        legend = 'on'
        legendloc = 'upper left'
        tick_disp = 'on'
        fit_line_style = '--'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5

    elif type == 'Kd_example':
        ylim = 'off'
        offset = 'off'
        bg_disp = 'off'
        fmax = 'full'
        kd = 'on'
        legend = 'off'
        legendloc = 'upper left'
        tick_disp = 'on'
        fit_line_style = '--'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5

    elif type == 'flowchartzoom':

        ylim = 'zoom'
        legend = 'off'
        tick_disp = 'off'
        offset = 'on'
        bg_disp = 'off'
        fmax = 'off'
        kd = 'off'
        legendloc = 'upper left'
        fit_line_style = '-'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5


    elif type == 'variance':

        ylim = 'variance'
        legend = 'off'
        tick_disp = 'on'
        offset = 'on'
        bg_disp = 'off'
        fmax = 'off'
        kd = 'off'
        legendloc = 'upper left'
        fit_line_style = '-'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5

    elif type == 'flowchart':

        ylim = 'on'
        legend = 'off'
        tick_disp = 'off'
        offset = 'on'
        bg_disp = 'off'
        fmax = 'off'
        kd = 'off'
        legendloc = 'upper left'
        fit_line_style = '-'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 5

    else:

        ylim = 'on'
        legend = 'on'
        tick_disp = 'on'
        offset = 'off'
        bg_disp = 'off'
        fmax = 'off'
        kd = 'off'
        legendloc = 'upper left'
        fit_line_style = '-'
        data_marker_style='o'
        data_marker_facecolor = 'none'
        data_marker_edgecolor = 'k'
        data_marker_size = 20


    ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                          (plot_location['gridpos_y'], plot_location['gridpos_x']), rowspan=plot_location['rowspan'],
                          colspan=plot_location['colspan'])
    if logplots == True:
        ax.scatter(np.log10(fig_data['x_data']), fig_data['y_data'], marker=data_marker_style, s=data_marker_size, facecolor=data_marker_facecolor, edgecolor=data_marker_edgecolor,
                   linewidths=0.5)
        xvalues = np.log10(y_fits['x_fit'])
        xlim_maxvalue = 3.2
        ax.set_xlim([-3.2, xlim_maxvalue])
        ax.set_xticks([-3,-2,-1,0,1,2,3])
        convert_ticks_to_log_format(ax, 'x')
    else:
        ax.scatter(fig_data['x_data'], fig_data['y_data'], marker=data_marker_style, s=data_marker_size, facecolor=data_marker_facecolor, edgecolor=data_marker_edgecolor,
                   linewidths=0.5)
        xvalues = y_fits['x_fit']
        if max(fig_data['x_data']) >= 10:
            xlim_maxvalue = 10.5
            ax.set_xlim([-0.5, xlim_maxvalue])
            ax.set_xticks([0, 2, 4, 6, 8, 10])
        else:
            xlim_maxvalue = 5.5
            ax.set_xlim([-0.5, xlim_maxvalue])
            ax.set_xticks([0,1,2,3,4,5])

    if bg_disp == 'on':
        ylim_minvalue = np.min([np.min(fig_data['y_data']),np.min(fig_data['bg'])]) * 0.95
    elif ylim == 'zoom':
        ylim_minvalue = np.min(fig_data['y_data'])+10
    else:
        ylim_minvalue = np.min([np.min(fig_data['y_data'])]) * 0.95

    if ylim == 'zoom':
        ylim_maxvalue = np.max(np.max(fig_data['y_data']))+2
    else:
        ylim_maxvalue = np.max([fig_data['bg'], np.max(fig_data['y_data']), fig_data['fo_Fmax']]) * 1.1

    if ylim == 'off':
        ax.set_ylim([ylim_minvalue*.9, ylim_maxvalue])

    if ylim == 'zoom':
        ax.set_ylim([ylim_minvalue*.9, ylim_maxvalue])

    if ylim == 'variance':
        ylim_minvalue = 175
        ylim_maxvalue = 325
        ax.set_ylim([ylim_minvalue, ylim_maxvalue])

    if fmax == 'full':
        ax.axhline(fig_data['fo_Fmax'],linestyle = '--', color='g', linewidth=0.5)
        if logplots == True:
            ax.text(xlim_maxvalue*1.05, fig_data['fo_Fmax']-5, '$F_{max}$', size=8, color='g')
            half_fmax = fig_data['fo_offset']+(fig_data['fo_Fmax']-fig_data['fo_offset'])/2
            ax.axhline(half_fmax, linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, half_fmax, '$\\frac{1}{2}$$F$$_{max}$', size=8, color='g')
            ax.axhline(fig_data['fo_offset'], linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, fig_data['fo_offset'], '$F_0$', size=8, color='g')
        else:
            ax.text(xlim_maxvalue*1.01,fig_data['fo_Fmax']-5,'$F_{max}$',size=8,color='g')
            half_fmax = fig_data['fo_offset']+(fig_data['fo_Fmax']-fig_data['fo_offset'])/2
            ax.axhline(half_fmax, linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, half_fmax, '$\\frac{1}{2}$$F$$_{max}$', size=8, color='g')
            ax.axhline(fig_data['fo_offset'], linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, fig_data['fo_offset'], '$F_0$', size=8, color='g')
    elif fmax == 'on':
        ax.axhline(fig_data['fo_Fmax'],linestyle = '--', color='g', linewidth=0.5)
        if logplots == True:
            ax.text(xlim_maxvalue*1.05, fig_data['fo_Fmax']-5, '$F_{max}$', size=8, color='g')
            ax.axhline(fig_data['fo_offset'], linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, fig_data['fo_offset'], '$F_0$', size=8, color='g')
        else:
            ax.text(xlim_maxvalue*1.01,fig_data['fo_Fmax']-5,'$F_{max}$',size=8,color='g')
            ax.axhline(fig_data['fo_offset'], linestyle='--', color='g', linewidth=0.5)
            ax.text(xlim_maxvalue * 1.01, fig_data['fo_offset'], '$F_0$', size=8, color='g')

    if kd == 'on':
        if logplots == True:
            ax.text(np.log10(fig_data['fo_fitKd']), ylim_maxvalue*1.01, '$K_d$', size=8, color='g')
            ax.axvline(np.log10(fig_data['fo_fitKd']), linestyle='--', color='g', linewidth=0.5)
        else:
            if fig_data['fo_fitKd'] < xlim_maxvalue:
                ax.text(fig_data['fo_fitKd'],ylim_maxvalue*1.01,'$K_d$',size=8,color='g')
                ax.axvline(fig_data['fo_fitKd'], linestyle='--', color='g', linewidth=0.5)

    if ylim == 'on':
        ax.set_ylim([100, 400])

    if offset == 'on':
        ax.plot(xvalues, [y_fits['fit_offset']]*len(xvalues), linestyle=':', color='gray', label='fit offset', linewidth=1)

    if bg_disp == 'on':
        ax.plot(xvalues, [fig_data['bg']]*len(xvalues), linestyle=':', color='orange', label='pub bg', linewidth=1)

    for i, fit_type in enumerate(fits_to_plot):
        if fit_type == 'fo':
            ax.plot(xvalues, y_fits['fo'], color='b', label='one-to-one', linewidth=0.5, linestyle = fit_line_style)
        if fit_type == 'linear':
            ax.plot(xvalues, y_fits['linear'], color='r', label='linear', linewidth=0.5,linestyle = fit_line_style)

    if ylabel:
        ax.set_ylabel(ylabel,size=6)

    if xlabel==True:
        if logplots == True:
            ax.set_xlabel('Concentration ($\mu$M, $log_{10}$)', size=6)
        else:
            ax.set_xlabel('Concentration ($\mu$M)', size=6)

    if Kd_label == True:
        half_Fmax =  y_fits['fit_offset'] + (y_fits['fit_Fmax'] / 2)
        if logplots == True:
            x_loc = np.log10(y_fits['fit_Kd'])
            #ax.plot([x_loc, x_loc], [0, half_Fmax], color='firebrick', linestyle='--', linewidth=0.5)
            offset = 0.3
        else:
            x_loc = y_fits['fit_Kd']
            #ax.plot([x_loc, x_loc], [0, half_Fmax], color='firebrick', linestyle='--', linewidth=0.5)
            offset = 0.5
        ax.text(x_loc+offset,half_Fmax*.9,'Kd = '+str(round(y_fits['fit_Kd'],2)),size=6,color='firebrick')

    if Kd_fixedlabel == True:
        if logplots == True:
            x_loc = np.log10(0)
            offset = 0.3
        else:
            x_loc = 0
            offset = 0.5
        ax.text(x_loc+offset,300,'Kd = '+str(round(y_fits['fit_Kd'],2)),size=6,color='firebrick')

    if title:
        ax.set_title(title,size=8)

    if legend == 'on':
        ax.legend(loc=legendloc, fontsize=5, frameon=False)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(5)
        #tick.label.set_rotation('vertical')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(5)

    if tick_disp == 'off':
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticks([])


def plot_domain_results(ax, rep_df, binding_field, xticks_on=True, ylabel_on=False, xlabel_on=False):
    # binding_field = 'binding_call' or 'binding_call_functionality_evaluated'

    for domain, domain_df in rep_df.reset_index().groupby(['domain']):
        if domain:

            plot_df = imperfect_merge(domain_df, domain, binding_field)

            # create plottable df
            binding_call_cols = [col for col in plot_df.columns if 'binding_call' in col]
            plot_df = plot_df[binding_call_cols]

            # 1000 is not measured
            if binding_field == 'binding_call':
                plot_df = plot_df.applymap(
                    lambda x: 0 if 'nobinding' in str(x) else 1 if 'binder' in str(x) else 0 if 'aggregator' in str(
                        x) else 0 if 'low-SNR' in str(x) else 1000)
            if binding_field == 'binding_call_functionality_evaluated':
                plot_df = plot_df.applymap(
                    lambda x: 0 if 'nobinding' in str(x) else 1 if 'binder' in str(x) else 0 if 'aggregator' in str(
                        x) else 0 if 'low-SNR' in str(x) else 4 if 'non-functional' in str(x) else 1000)

            subplot_heatmap_colormesh(plot_df, domain, ax, xticks_on)
            if xlabel_on == True:
                ax.set_xlabel('Run', size=6)
            if ylabel_on == True:
                ax.set_ylabel('Peptide', size=6)


def plot_Kd_scatter(ax, df,kD_threshold, xlabel_on=False, ylabel_on=False):

    df = df[(df.fit_Kd <= kD_threshold) & (df.pub_Kd_mean <= kD_threshold)]
    df = df.fillna(0)
    xdata = df['fit_Kd'].values
    ydata = df['pub_Kd_mean'].values

    pearson_r = stats.pearsonr(xdata, ydata)[0]

    plt.scatter(xdata, ydata, marker='o', s=10, facecolor='none', edgecolor='b', linewidths=0.2, rasterized=True)
    if xlabel_on == True:
        ax.set_xlabel(r'Revised Fits ($\mu$M)',size=6)
    if ylabel_on == True:
        ax.set_ylabel(r'Published Fits ($\mu$M)',size=6)

    if kD_threshold > 1:
        ax.set_xlim([-0.2, kD_threshold*1.05])
        ax.set_ylim([-0.2, kD_threshold*1.05])
    else:
        ax.set_xlim([-0.05, kD_threshold*1.05])
        ax.set_ylim([-0.05, kD_threshold*1.05])

    ax.text(kD_threshold*.6, kD_threshold/8, 'Pearson r = ' + str.format('{:.3f}', pearson_r) + '\non ' + str(df.shape[0]) + ' points', size=6)
    ax.set_title('Threshold $\leq$'+str(kD_threshold)+'$\mu$M',size=8)


def boxplot_sorted(df, by, column, sort_by_median=True,rot=0,fs=10):
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    meds = df2.median().sort_values()
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    if sort_by_median == True:
        return df2[meds.index].boxplot(rot=rot, return_type="axes",fontsize=fs,showfliers=False)
    else:
        return df2.boxplot(rot=rot, return_type="axes",fontsize=fs,showfliers=False)


def calculate_delta_Kd(df):
    df['delta_Kd']=np.nan
    # calculate delta values

    # CASES where I want to plot a zero delta value:
    # when I report non-binding or low-SNR or aggregator, and they report nan (>20uM Kd), record no change
    # when I report Kd>20uM, and they report nan (>20uM Kd), record no change
    # when we both are missing values, record no change
    df.loc[(df.functional_binding_call == 'no-binding') & (df.pub_Kd_mean.isnull()), 'delta_Kd'] = 0
    df.loc[(df.functional_binding_call == 'low-SNR') & (df.pub_Kd_mean.isnull()), 'delta_Kd'] = 0
    df.loc[(df.functional_binding_call == 'aggregator') & (df.pub_Kd_mean.isnull()), 'delta_Kd'] = 0
    df.loc[(df.functional_binding_call == 'mixed_agg_lowsnr_nonfunc') & (df.pub_Kd_mean.isnull()), 'delta_Kd'] = 0
    df.loc[(df.functional_binding_call == 'nonfunctional') & (df.pub_Kd_mean.isnull()), 'delta_Kd'] = 0
    df.loc[(df.functional_binding_call == 'binder') & (df.fit_Kd > 20) & (df.pub_Kd_mean.isnull()), 'delta_Kd'] = 0
    df.loc[(df.fit_Kd.isnull())&(df.pub_Kd_mean.isnull()),'delta_Kd'] = 0

    # CASES where I want to plot a non-zero delta value:
    # when I report non-binding, low-SNR, or aggregator and they report <20uM Kd, record 20-PUB_Kd_mean
    # when I report Kd<20uM, and they report nan (>20uM Kd), record 20-MY_Kd
    df.loc[(df.functional_binding_call == 'no-binding') & ~(df.pub_Kd_mean.isnull()),'delta_Kd'] = 20 - df[(df.functional_binding_call == 'no-binding') & ~(df.pub_Kd_mean.isnull())].pub_Kd_mean
    df.loc[(df.functional_binding_call == 'low-SNR') & ~(df.pub_Kd_mean.isnull()),'delta_Kd'] = 20 - df[(df.functional_binding_call == 'low-SNR') & ~(df.pub_Kd_mean.isnull())].pub_Kd_mean
    df.loc[(df.functional_binding_call == 'aggregator') & ~(df.pub_Kd_mean.isnull()),'delta_Kd'] = 20 - df[(df.functional_binding_call == 'aggregator') & ~(df.pub_Kd_mean.isnull())].pub_Kd_mean
    df.loc[(df.functional_binding_call == 'mixed_agg_lowsnr_nonfunc') & ~(df.pub_Kd_mean.isnull()),'delta_Kd'] = 20 - df[(df.functional_binding_call == 'low-SNR') & ~(df.pub_Kd_mean.isnull())].pub_Kd_mean
    df.loc[(df.functional_binding_call == 'nonfunctional') & ~(df.pub_Kd_mean.isnull()),'delta_Kd'] = 20 - df[(df.functional_binding_call == 'aggregator') & ~(df.pub_Kd_mean.isnull())].pub_Kd_mean
    df.loc[(df.functional_binding_call == 'binder') & (df.fit_Kd <= 20)&(df.pub_Kd_mean.isnull()),'delta_Kd'] = 20 - df[(df.functional_binding_call == 'binder') & (df.pub_Kd_mean.isnull())].fit_Kd
    df.loc[(df.functional_binding_call.isnull()) & ~(df.pub_Kd_mean.isnull()),'delta_Kd'] = 20 - df[(df.functional_binding_call.isnull()) & ~(df.pub_Kd_mean.isnull())].pub_Kd_mean


    # all other cases are straight subtraction
    df.loc[df.delta_Kd.isnull(),'delta_Kd'] = df[df.delta_Kd.isnull()].fit_Kd - df[df.delta_Kd.isnull()].pub_Kd_mean

    return df


def nonzerodist(u, v):
    new_u = []
    new_v = []
    nonbinding_penalty = 0
    nonbinding_value = 0.5
    for u_item, v_item in zip(u, v):
        if u_item <=20 or v_item <=20:
            if u_item <=20:
                new_u.append(u_item)
            else:
                new_u.append(21)
            if v_item <=20:
                new_v.append(v_item)
            else:
                new_v.append(21)
        else:
            nonbinding_penalty += nonbinding_value
    if new_u == [] or new_v == []:
        return nonbinding_value*len(u)
    else:
        new_u = np.asarray(new_u)
        new_v = np.asarray(new_v)

        return nonbinding_penalty + np.sqrt(((new_u-new_v)**2).sum())


def merge_sorted_lists(list_of_lists):
    master_index_list=[]
    for i,new_list in enumerate(list_of_lists):
        location_of_last_match = None
        found_any_duplicate_flag = False
        for pepstr in new_list:
            if  master_index_list==[]: # if list is empty, just append first one
                master_index_list.append(pepstr)
            elif pepstr in master_index_list:# if this peptide is a duplicate of one in the list already, update location
                found_any_duplicate_flag = True
                location_of_last_match = master_index_list.index(pepstr)+1 # record its position in master list
                pass
            else: # if list not empty, and this is not a dup, then insert at appropriate place in list
                if i>0 and found_any_duplicate_flag: # if at least one peptide from the list matched the master already
                    #insert it after that location
                    master_index_list = master_index_list[:location_of_last_match]+[pepstr]+master_index_list[location_of_last_match:]
                else: #add at the end
                    master_index_list.append(pepstr)
    return master_index_list


def imperfect_merge(df, domain_name, binding_field):
    tomerge_lists = []
    for expt_run, expt_df in df.reset_index().groupby(['filename']):
        pep_list = expt_df.peptide_str.values.tolist()
        tomerge_lists.append(pep_list)
    mlist = merge_sorted_lists(tomerge_lists)
    left_list = [(domain_name, x, i) for i, x in enumerate(mlist)]
    left_df = pd.DataFrame(left_list, columns=['domain', 'peptide_str', 'order'])
    expt_list = df.filename.unique()

    expt_df_list = []
    for expt in expt_list:
        merge_df = df[df.filename == expt].copy()
        merge_df = merge_df[['domain', 'peptide_str', 'plate_number', 'domain_pos', 'fit_Kd', binding_field]]
        merge_df = merge_df.groupby(['domain', 'peptide_str']).agg(lambda x: tuple(x)).reset_index()
        merge_df = merge_df.add_prefix(expt + '_')
        expt_df = pd.merge(left_df, merge_df, how='left', left_on=['domain', 'peptide_str'],
                           right_on=[expt + '_' + 'domain', expt + '_' + 'peptide_str'])
        expt_df_list.append(expt_df)
    final_domain_df = pd.concat([x.set_index(['domain', 'peptide_str', 'order']) for x in expt_df_list], axis=1)
    for expt in expt_list:
        final_domain_df = final_domain_df.drop([expt + '_domain', expt + '_peptide_str'], axis=1)

    return final_domain_df


def subplot_heatmap_colormesh(df, domain, ax, xticks_on=True):
    data = df.values
    cols = {0: 'white', 1: 'green', 2: 'darkorange', 3: 'dimgray', 4: 'dodgerblue', 1000: 'lightgray'}

    cvr = colors.ColorConverter()
    tmp = sorted(cols.keys())
    cols_rgb = [cvr.to_rgb(cols[k]) for k in tmp]
    intervals = np.array([tmp[0]] + [x + 0.5 for x in tmp])
    cmap, norm = colors.from_levels_and_colors(intervals, cols_rgb)
    pcm = ax.pcolormesh(data, cmap=cmap, norm=norm, linewidth=0.5)
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    x_axis_labels = np.arange(len(df.columns.values)) + 1
    ax.set_xticks(np.arange(len(x_axis_labels)) + 0.5)
    if xticks_on:
        ax.set_xticklabels(x_axis_labels, size=4)
    else:
        ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_ylim([0, df.shape[0]])
    ax.set_title(domain, y=0.95, size=6)


def get_upper_triangle_index(fig_qty):
    # create upper triangle figure index
    fig_idx = []
    for n in np.arange(fig_qty):
        fig_idx.extend((n*fig_qty+1)+np.arange(fig_qty)[n:])
    return fig_idx


from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fig_func_lib import plotting_functions as pf
import ms_func_lib.Kd_fitting_functions as cff
from matplotlib.backends.backend_pdf import PdfPages
import math
from matplotlib import colors
import scipy.cluster.hierarchy as sch
from scipy import stats
import itertools as it


def report_allplots(print_data, output_file_name):
    page_count = 0
    for i, row in enumerate(print_data.itertuples()):
        # if print document hits 1000 pages, create new document and rename
        if page_count == 0:
            pdf_pages = PdfPages(output_file_name + '_' + str(page_count).zfill(5) + '.pdf')
        elif page_count % 1000 == 0:
            pdf_pages.close()
            pdf_pages = PdfPages(output_file_name + '_' + str(page_count).zfill(5) + '.pdf')

        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(row.domain+' '+row.gene_name+' '+str(row.pY_pos))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        # Table
        plot_location = pf.set_plot_location(4, 16, 0, 0, 1, 4)
        pf.plot_table(row, plot_location, 'id')

        # Table
        plot_location = pf.set_plot_location(4, 16, 0, 1, 1, 4)
        pf.plot_table(row, plot_location, 'chosenfit')

        # Table
        plot_location = pf.set_plot_location(4, 16, 0, 2, 2, 4)
        pf.plot_table(row, plot_location, 'fulldata')

        # Figure
        plot_location = pf.set_plot_location(4, 3, 1, 0)
        pf.plot_Kd_curves_v2(['linear', 'fo'], fig_data, yfits, plot_location, fig_data['title'], bg_disp='on')

        # Figure
        plot_location = pf.set_plot_location(4, 3, 1, 1)
        pf.plot_Kd_curves_v2(['linear', 'fo'], fig_data, yfits, plot_location, fig_data['title'], logplots=True, bg_disp='on')

        # Figure
        plot_location = pf.set_plot_location(4, 3, 2, 0)
        pf.plot_Kd_curves_v2(['linear', 'fo'], fig_data, yfits, plot_location, fig_data['title'], ylim='off', bg_disp='on')

        # Figure
        plot_location = pf.set_plot_location(4, 3, 2, 1)
        pf.plot_Kd_curves_v2(['linear', 'fo'], fig_data, yfits, plot_location, fig_data['title'], logplots=True, ylim='off', bg_disp='on')


        # Figure
        plot_location = pf.set_plot_location(4, 3, 2, 2)
        pf.plot_residuals(fig_data, yfits, plot_location, logplots=True)


        # save page to pdf i
        pdf_pages.savefig(fig)
        page_count += 1
        fig.clf()
        plt.close()

    fig.clf()
    plt.close()
    pdf_pages.close()


def report_groupedplots(print_data, group_cols, sort_columns, output_file_name):
    print_data_grouped = print_data.reset_index().sort_values(sort_columns).groupby(group_cols)
    page_count = 0

    for replicate_group_description, replicate_group_df in print_data_grouped:
        # if print document hits 1000 pages, create new document and rename
        if page_count == 0:
            pdf_pages = PdfPages(output_file_name + '_' + str(page_count).zfill(5) + '.pdf')
        elif page_count % 1000 == 0:
            pdf_pages.close()
            pdf_pages = PdfPages(output_file_name + '_' + str(page_count).zfill(5) + '.pdf')
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(str(replicate_group_description))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        fig_columns = 3
        fig_rows = 4
        fig_rows_with_table = fig_rows - 1
        max_figures_per_page = fig_columns * fig_rows
        max_figures_per_page_with_table = fig_columns * fig_rows_with_table


        if replicate_group_df.shape[0] <= max_figures_per_page_with_table:
            # Table
            plot_location = pf.set_plot_location(4, 16, 0, 0, 2, 4)
            pf.plot_table(replicate_group_df, plot_location, 'replicates')

            for i, row in enumerate(replicate_group_df.itertuples()):
                fig_indexer = i

                fit_parameters = pf.set_fit_parameters(row)
                fig_data = pf.get_fig_data(row)
                fig_title = fig_data['filename'] +'|'+str(fig_data['plate_number']) +'|'+str(fig_data['domain_pos'])
                yfits = pf.generate_yfits(fig_data, fit_parameters)

                # Figure
                plot_location = pf.set_plot_location(fig_rows, fig_columns, 1 + int(math.floor(fig_indexer / fig_columns)), fig_indexer % fig_columns, 1, 1)

                pf.plot_Kd_curves_v2(['linear', 'fo'], fig_data, yfits, plot_location, fig_title, ylim='off')
        else:
            # Table
            plot_location = pf.set_plot_location(4, 16, 0, 0, 16, 4)
            pf.plot_table(replicate_group_df, plot_location, 'replicates')

            for i, row in enumerate(replicate_group_df.itertuples()):
                fig_indexer = i % max_figures_per_page
                if fig_indexer == 0:
                    # start new page
                    pdf_pages.savefig(fig)
                    page_count += 1
                    fig.clf()
                    plt.close()
                    fig = plt.figure(figsize=(8.5, 11))
                    fig.suptitle(str(replicate_group_description) + ' <continued>')
                    plt.subplots_adjust(hspace=0.3, wspace=0.3)

                fit_parameters = pf.set_fit_parameters(row)
                fig_data = pf.get_fig_data(row)
                fig_title = fig_data['filename'] +'|'+str(fig_data['plate_number']) +'|'+str(fig_data['domain_pos'])
                yfits = pf.generate_yfits(fig_data, fit_parameters)

                # Figure
                plot_location = pf.set_plot_location(fig_rows, fig_columns, int(math.floor(fig_indexer / fig_columns)), fig_indexer % fig_columns, 1, 1)
                pf.plot_Kd_curves_v2(['linear', 'fo'], fig_data, yfits, plot_location, fig_title, ylim='off')


        # save page to pdf i
        pdf_pages.savefig(fig)
        page_count += 1
        fig.clf()
        plt.close()

    fig.clf()
    plt.close()

    pdf_pages.close()


def report_distributions_KdFmax_v2(df, outfile_name, fit_type=None):
    fig = plt.figure(figsize=(6.5, 6.5))
    fig.suptitle('Parameter Distribution\nfor Saturated Interactions', size=12)

    # df = df.dropna(subset = ['fit_Kd', 'fit_Fmax'])
    if fit_type == 'force fo':
        logFmax_data = np.log10(df.fo_Fmax)
        logKd_data = np.log10(df.fo_Kd)
    else:
        logFmax_data = np.log10(df.fit_Fmax)
        logKd_data = np.log10(df.fit_Kd)

    figs_tall = 1
    figs_wide = 1
    fontsize = 10
    labelsize = 7

    ax = plt.subplot(figs_tall, figs_wide, 1)

    # log Fmax vs log Kd
    plt.scatter(logFmax_data, logKd_data, color='b', marker='.', s=2, rasterized=True)
    ax.set_xlabel('$F_{max}$', fontsize=fontsize)
    ax.set_ylabel('$K_D$', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize)

    Fmax_tick_array = np.asarray([10, 25, 50, 100, 150, 200, 250, 300, 350])
    ax.set_xticks(np.log10(Fmax_tick_array))
    ax.set_xticklabels(Fmax_tick_array)

    Kd_tick_array = np.asarray([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5])
    ax.set_yticks(np.log10(Kd_tick_array))
    ax.set_yticklabels(Kd_tick_array)

    fig.savefig(outfile_name + '.pdf', dpi=600, bbox_inches='tight')
    #fig.savefig(outfile_name + '.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def report_domain_boxplots_Fmax(df, outfile_name, sort_by_median=True):
    result_list = []

    df = df.reset_index()
    fig = plt.figure(figsize=(8, 6.5))
    axes = pf.boxplot_sorted(df.reset_index(), 'domain', 'fit_Fmax', sort_by_median, rot=90, fs=6)
    axes.set_title('$F_{max}$ Distributions by Domain')
    fig.savefig(outfile_name + '.pdf', dpi=600, bbox_inches='tight')
    #fig.savefig(outfile_name + '.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def report_grouped_boxplots(df, group_by_cols, plot_col, outfile_name, sort_by_median=True):
    result_list = []

    df = df.reset_index()
    fig = plt.figure(figsize=(8, 6.5))
    axes = pf.boxplot_sorted(df.reset_index(), group_by_cols, plot_col, sort_by_median, rot=90, fs=6)
    # axes.set_title('Distributions: '+plot_col+' grouped by '+group_by_cols)
    axes.set_title('$F_{max}$ Distributions by Peptide')
    fig.savefig(outfile_name + '.pdf', dpi=600, bbox_inches='tight')
    #fig.savefig(outfile_name + '.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def report_results_heatmap_colormesh(df, filename,pivot_artifact=False):
    # following rows needed to start in upper left, correct axis label order, and plot in correct orientation
    df = df.transpose()
    df = df.reindex(index=df.index[::-1])
    data = df.values
    peploc_list = df.index.values.tolist()
    domain_list = df.columns.tolist()
    ################################################
    fig = plt.figure(figsize=(6.5, 8))
    axmatrix = fig.add_axes([0.06, 0.05, 0.85, 0.94])
    if pivot_artifact == False:
        cols = {0: 'white', 0.5: 'yellow', 1: 'orange', 2: 'red', 5: 'darkred', 20: 'black', 1000: 'steelblue',
                1002: 'darkgray'}
    else:
        cols = {0: 'white', 0.5: 'yellow', 1: 'orange', 2: 'red', 5: 'darkred', 20: 'black', 1000: 'steelblue',
                1001: 'darkgray', 1002: 'sienna'}


    cvr = colors.ColorConverter()
    tmp = sorted(cols.keys())
    cols_rgb = [cvr.to_rgb(cols[k]) for k in tmp]
    intervals = np.array(tmp + [tmp[-1] + 1])
    cmap, norm = colors.from_levels_and_colors(intervals, cols_rgb)
    pcm = axmatrix.pcolormesh(data, cmap=cmap, norm=norm, linewidth=0.5, edgecolors='gray')

    axmatrix.set_yticks(np.arange(len(peploc_list)) + 0.5)
    axmatrix.set_yticklabels(peploc_list, size=3)
    axmatrix.set_ylim([0, len(peploc_list)])

    axmatrix.set_xticks(
        np.arange(len(domain_list)) + 0.5)  # note these changes to xticks necessary to get labels aligned
    axmatrix.set_xticklabels(domain_list, rotation=90, size=3)
    axmatrix.set_xlim([0, len(domain_list)])  # note these changes to xticks necessary to get labels aligned

    axmatrix.tick_params(axis='both', which='major', labelsize=4)

    # Plot colorbar.
    axcolor = fig.add_axes([0.92, 0.25, 0.02, 0.50])
    cbar = plt.colorbar(pcm, cax=axcolor)
    axcolor.tick_params(labelsize=6)
    if pivot_artifact == False:
        cbar.set_ticks([0.25, 0.75, 1.5, 3.5, 12.5, 500, 1001, 1002.5])
        cbar.set_ticklabels(
            ['<0.5 uM', '0.5 to\n1.0 uM', '1.0 to\n2.0 uM', '2.0 to\n5.0 uM', '5 to\n20 uM', '>20 uM', 'non-funct',
             'not meas'])
    else:
        cbar.set_ticks([0.25, 0.75, 1.5, 3.5, 12.5, 500, 1000.5, 1001.5, 1002.5])
        cbar.set_ticklabels(
            ['<0.5 uM', '0.5 to\n1.0 uM', '1.0 to\n2.0 uM', '2.0 to\n5.0 uM', '5 to\n20 uM', '>20 uM', 'non-funct',
             'not meas','pivot artifact'])

    fig.savefig(filename + '.pdf', bbox_inches='tight')
    #fig.savefig(filename + '.eps', bbox_inches='tight')
    plt.close('all')


def report_results_heatmap_bindingcallchanges(df, filename):
    # following rows needed to start in upper left, correct axis label order, and plot in correct orientation
    df = df.transpose()
    df = df.reindex(index=df.index[::-1])
    data = df.values
    peploc_list = df.index.values.tolist()
    domain_list = df.columns.tolist()
    ################################################
    fig = plt.figure(figsize=(6.5, 8))
    axmatrix = fig.add_axes([0.06, 0.05, 0.85, 0.94])
    cols = {0: 'white', 1: 'darkgray', 2: 'orange', 3: 'dodgerblue', 4: 'lightgray'}


    cvr = colors.ColorConverter()
    tmp = sorted(cols.keys())
    cols_rgb = [cvr.to_rgb(cols[k]) for k in tmp]
    intervals = np.array(tmp + [tmp[-1] + 1])
    cmap, norm = colors.from_levels_and_colors(intervals, cols_rgb)
    pcm = axmatrix.pcolormesh(data, cmap=cmap, norm=norm, linewidth=0.5, edgecolors='gray')

    axmatrix.set_yticks(np.arange(len(peploc_list)) + 0.5)
    axmatrix.set_yticklabels(peploc_list, size=3)
    axmatrix.set_ylim([0, len(peploc_list)])

    axmatrix.set_xticks(
        np.arange(len(domain_list)) + 0.5)  # note these changes to xticks necessary to get labels aligned
    axmatrix.set_xticklabels(domain_list, rotation=90, size=3)
    axmatrix.set_xlim([0, len(domain_list)])  # note these changes to xticks necessary to get labels aligned

    axmatrix.tick_params(axis='both', which='major', labelsize=4)

    # Plot colorbar.
    axcolor = fig.add_axes([0.92, 0.25, 0.02, 0.50])
    cbar = plt.colorbar(pcm, cax=axcolor)
    axcolor.tick_params(labelsize=6)
    cbar.set_ticks([0.5,1.5,2.5,3.5,4.5])
    cbar.set_ticklabels(['Other', 'Binder', 'Publication: Binder\nRevised: Non-Binder', 'Publication: Rejected\nRevised: Binder', 'Non-Binder'])

    fig.savefig(filename + '.pdf', bbox_inches='tight')
    #fig.savefig(filename + '.eps', bbox_inches='tight')
    plt.close('all')


def report_results_deltaheatmap_colormesh(df, filename):
    # following rows needed to start in upper left, correct axis label order, and plot in correct orientation
    df = df.transpose()
    df = df.reindex(index=df.index[::-1])
    data = df.values
    peploc_list = df.index.values.tolist()
    domain_list = df.columns.tolist()
    ################################################
    fig = plt.figure(figsize=(6.5, 8))
    axmatrix = fig.add_axes([0.06, 0.05, 0.85, 0.94])

    # cols = {0:'black',0.5:'darkred',1:'red',2:'orange',5:'yellow',20:'white',1000:'gray'}
    # cvr = colors.ColorConverter()
    # tmp = sorted(cols.keys())
    # cols_rgb = [cvr.to_rgb(cols[k]) for k in tmp]
    # intervals = np.array(tmp + [tmp[-1]+1])
    # cmap, norm = colors.from_levels_and_colors(intervals,cols_rgb)
    # pcm = axmatrix.pcolormesh(data,cmap = cmap, norm = norm,linewidth=0.5,edgecolors='gray')
    pcm = axmatrix.pcolormesh(data, cmap='bwr', linewidth=0.5, edgecolors='gray', vmin=-20, vmax=20)

    axmatrix.set_yticks(np.arange(len(peploc_list)) + 0.5)
    axmatrix.set_yticklabels(peploc_list, size=3)
    axmatrix.set_ylim([0, len(peploc_list)])

    axmatrix.set_xticks(
        np.arange(len(domain_list)) + 0.5)  # note these changes to xticks necessary to get labels aligned
    axmatrix.set_xticklabels(domain_list, rotation=90, size=3)
    axmatrix.set_xlim([0, len(domain_list)])  # note these changes to xticks necessary to get labels aligned

    axmatrix.tick_params(axis='both', which='major', labelsize=4)

    # Plot colorbar.
    axcolor = fig.add_axes([0.92, 0.25, 0.02, 0.50])
    cbar = plt.colorbar(pcm, cax=axcolor)
    axcolor.tick_params(labelsize=6)
    # cbar.set_ticks([0.25,0.75,1.5,3.5,12.5,500,1000.5])
    # cbar.set_ticklabels(['<0.5 uM','0.5 to\n1.0 uM','1.0 to\n2.0 uM','2.0 to\n5.0 uM','5 to\n20 uM','>20 uM','n/a'])

    fig.savefig(filename + '.pdf', bbox_inches='tight')
    #fig.savefig(filename + '.eps', bbox_inches='tight')
    plt.close('all')


def report_results_heatmap_w_dendrogram_colormesh(df, filename,pivot_artifact=False):
    # following rows needed to start in upper left, correct axis label order, and plot in correct orientation
    df = df.transpose()
    #df = df.reindex(index=df.index[::-1])
    data = df.values
    peploc_list = df.index.values.tolist()
    domain_list = df.columns.tolist()
    ################################################
    fig = plt.figure(figsize=(6.5, 8))

    ax_xdend = fig.add_axes([.14, 0, .76, 0.07])
    ax_xdend.axis('off')
    D = data.T
    Yx = sch.linkage(D, method='average',metric=pf.nonzerodist)
    with plt.rc_context({'lines.linewidth': 0.5}):
        Zx = sch.dendrogram(Yx, orientation='bottom', labels=domain_list, color_threshold = 2)

    ax_ydend = fig.add_axes([0, 0.12, 0.07, 0.78])
    ax_ydend.axis('off')
    D = data
    Yy = sch.linkage(D, method='average',metric=pf.nonzerodist)
    with plt.rc_context({'lines.linewidth': 0.5}):
        Zy = sch.dendrogram(Yy, orientation='left', labels=peploc_list, color_threshold = 2)

    axmatrix = fig.add_axes([0.14, 0.12, 0.76, 0.78])
    if pivot_artifact == False:
        cols = {0: 'white', 0.5: 'yellow', 1: 'orange', 2: 'red', 5: 'darkred', 20: 'black', 1000: 'steelblue',
                1002: 'darkgray'}
    else:
        cols = {0: 'white', 0.5: 'yellow', 1: 'orange', 2: 'red', 5: 'darkred', 20: 'black', 1000: 'steelblue',
                1001: 'darkgray', 1002: 'sienna'}

    cvr = colors.ColorConverter()
    tmp = sorted(cols.keys())
    cols_rgb = [cvr.to_rgb(cols[k]) for k in tmp]
    intervals = np.array(tmp + [tmp[-1] + 1])
    cmap, norm = colors.from_levels_and_colors(intervals, cols_rgb)

    idxX = Zx['leaves']
    idxY = Zy['leaves']
    sorted_data = data[idxY, :]
    sorted_data = sorted_data[:, idxX]
    sorted_pep_labels = np.asarray(peploc_list)[idxY]
    sorted_dom_labels = np.asarray(domain_list)[idxX]

    pcm = axmatrix.pcolormesh(sorted_data, cmap=cmap, norm=norm, linewidth=0.5, edgecolors='gray')

    axmatrix.set_yticks(np.arange(len(sorted_pep_labels)) + 0.5)
    axmatrix.set_yticklabels(sorted_pep_labels, size=3)
    axmatrix.set_ylim([0, len(sorted_pep_labels)])

    axmatrix.set_xticks(
        np.arange(len(sorted_dom_labels)) + 0.5)  # note these changes to xticks necessary to get labels aligned
    axmatrix.set_xticklabels(sorted_dom_labels, rotation=90, size=3)
    axmatrix.set_xlim([0, len(sorted_dom_labels)])  # note these changes to xticks necessary to get labels aligned

    axmatrix.tick_params(axis='both', which='major', labelsize=4)

    # Plot colorbar.
    axcolor = fig.add_axes([0.92, 0.25, 0.02, 0.50])
    cbar = plt.colorbar(pcm, cax=axcolor)
    axcolor.tick_params(labelsize=6)
    if pivot_artifact == False:
        cbar.set_ticks([0.25, 0.75, 1.5, 3.5, 12.5, 500, 1001, 1002.5])
        cbar.set_ticklabels(
            ['<0.5 uM', '0.5 to\n1.0 uM', '1.0 to\n2.0 uM', '2.0 to\n5.0 uM', '5 to\n20 uM', '>20 uM', 'non-funct',
             'not meas'])
    else:
        cbar.set_ticks([0.25, 0.75, 1.5, 3.5, 12.5, 500, 1000.5, 1001.5, 1002.5])
        cbar.set_ticklabels(
            ['<0.5 uM', '0.5 to\n1.0 uM', '1.0 to\n2.0 uM', '2.0 to\n5.0 uM', '5 to\n20 uM', '>20 uM', 'non-funct',
             'not meas','pivot artifact'])

    fig.savefig(filename + '.pdf', bbox_inches='tight')
    #fig.savefig(filename + '.eps', bbox_inches='tight')
    plt.close('all')


def report_results_heatmap_colormesh_KMN(df, filename, figuresize=(8.5, 11), label_fontsize=5):
    ## Heatmap for KMN of EGFR tails and a subset of domains

    # following rows needed to start in upper left, correct axis label order, and plot in correct orientation
    df = df.transpose()
    df = df.reindex(index=df.index[::-1])
    data = df.values
    peploc_list = df.index.values.tolist()
    domain_list = df.columns.tolist()
    ################################################
    fig = plt.figure(figsize=(figuresize))
    axmatrix = fig.add_axes([0.06, 0.05, 0.85, 0.94])

    cols = {0: 'white', 0.5: 'yellow', 1: 'orange', 2: 'red', 5: 'darkred', 20: 'black', 1000: 'gray'}

    cvr = colors.ColorConverter()
    tmp = sorted(cols.keys())
    cols_rgb = [cvr.to_rgb(cols[k]) for k in tmp]
    intervals = np.array(tmp + [tmp[-1] + 1])
    cmap, norm = colors.from_levels_and_colors(intervals, cols_rgb)
    pcm = axmatrix.pcolormesh(data, cmap=cmap, norm=norm, linewidth=0.5, edgecolors='gray')

    axmatrix.set_yticks(np.arange(len(peploc_list)) + 0.5)
    axmatrix.set_yticklabels(peploc_list, size=5)
    axmatrix.set_ylim([0, len(peploc_list)])

    axmatrix.set_xticks(
        np.arange(len(domain_list)) + 0.5)  # note these changes to xticks necessary to get labels aligned
    axmatrix.set_xticklabels(domain_list, rotation=90, size=label_fontsize)
    axmatrix.set_xlim([0, len(domain_list)])  # note these changes to xticks necessary to get labels aligned

    axmatrix.tick_params(axis='both', which='major', labelsize=label_fontsize)
    axmatrix.tick_params(axis='both', which='minor', labelsize=label_fontsize)

    # Plot colorbar.
    # axcolor = fig.add_axes([0.92,0.25,0.02,0.50])
    axcolor = fig.add_axes([0.94, 0.05, 0.08, 0.66])
    cbar = plt.colorbar(pcm, cax=axcolor)
    axcolor.tick_params(labelsize=label_fontsize)
    cbar.set_ticks([0.25, 0.75, 1.5, 3.5, 12.5, 500, 1000.5])
    cbar.set_ticklabels(
        ['<0.5 $\mu$M', '0.5 to 1.0 $\mu$M', '1.0 to 2.0 $\mu$M', '2.0 to 5.0 $\mu$M', '5 to 20 $\mu$M', '>20 $\mu$M',
         'n/a'])

    fig.savefig(filename + '.pdf', bbox_inches='tight')
    #fig.savefig(filename + '.eps', bbox_inches='tight')
    plt.close('all')


def report_one_to_one_plot(filename,figuresize,label_fontsize):
    ## One to One Plot, Theoretical 1uM
    fig = plt.figure(figsize=figuresize)
    Kd = 1
    Fmax = 100
    offset = 0
    conc = np.arange(0,10,0.001)
    y_vals = cff.model_firstorder_offset(conc, Fmax, Kd, offset)

    ax = plt.subplot(111)
    ax.plot(conc,y_vals,color='k')
    ax.set_xlim([-0.1,10.1])
    ax.set_ylim([0,110])
    ax.set_xlabel('[concentration]')
    ax.set_yticks([50,100])
    ax.set_yticklabels(['$\\frac{1}{2}$$F$$_{max}$','$F$$_{max}$'],size=label_fontsize)
    ax.set_xticks([0,2,4,6,8,10])
    ax.set_xticklabels(['0','2','4','6','8','10'],size=label_fontsize)

    ax.plot([0,10],[100,100],color='g', linestyle='--', linewidth=1)
    ax.plot([1,1],[0,50],color='dodgerblue', linestyle='--', linewidth=1)
    ax.plot([0,1],[50,50],color='dodgerblue', linestyle='--', linewidth=1)
    plt.text(1.2,20,'$K_d$ =  1$\mu$$M$',color = 'k',size=8)

    ax.set_xlabel('Protein Concentration ($\mu$$M$)',size=8)
    fig.savefig(filename+'.pdf',bbox_inches='tight')
    #fig.savefig(filename+'.eps',bbox_inches='tight')
    plt.close('all')


def report_scatter_plots_dataset_comparison(df,filename):
    ## Scatter plots for comparison of JonesFP Original Calls vs Re-Fit Calls


    fig = plt.figure(figsize=(7.6, 7.6))
    #fig.suptitle('Published Affinity vs. Revised Affinity')
    figures_wide = 2
    figures_tall = 2
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, Kd_threshold in enumerate([40,20,5,1],start=1):
        ax = plt.subplot(figures_wide, figures_tall, i)

        data_cols = ['functional_binding_call', 'fit_Kd', 'pub_Kd_mean']
        binding_df = df.copy()
        binding_df = binding_df[data_cols]
        binding_df = binding_df[~((binding_df.fit_Kd.isnull()) & (binding_df.pub_Kd_mean.isnull()))]
        binding_df = binding_df[(binding_df.fit_Kd <= Kd_threshold) & (binding_df.pub_Kd_mean <= Kd_threshold)]
        if i == 1:
            pf.plot_Kd_scatter(ax,binding_df,Kd_threshold,ylabel_on=True)
        elif i == 3:
            pf.plot_Kd_scatter(ax,binding_df,Kd_threshold,xlabel_on=True,ylabel_on=True)
        elif i == 4:
            pf.plot_Kd_scatter(ax,binding_df,Kd_threshold,xlabel_on=True)
        else:
            pf.plot_Kd_scatter(ax,binding_df,Kd_threshold)


    fig.savefig('figures/'+filename+'.pdf', bbox_inches='tight', dpi=600)
    #fig.savefig('figures/'+filename+'.eps', bbox_inches='tight', dpi=600)
    plt.close('all')


def report_replicate_variance(replicate_df,grouped_df,output_filename):

    replicate_df['name']=replicate_df['domain'].astype(str) +'_'+ replicate_df['gene_name'].astype(str) +'_'+ replicate_df['pY_pos'].astype(str)
    replicate_df = replicate_df[replicate_df.binding_call_functionality_evaluated=='binder']

    grouped_df = grouped_df[~grouped_df['domain'].str.contains('GST')]
    grouped_df['name'] = grouped_df['domain'].astype(str) +'_'+ grouped_df['gene_name'].astype(str) +'_'+ grouped_df['pY_pos'].astype(str)

    binder_df = grouped_df[(grouped_df.functional_binding_call=='binder') & (grouped_df.count_bind>1) & (grouped_df.fit_Kd<=1)]
    binder_idx = binder_df['name'].drop_duplicates().reset_index(drop=True).apply(tuple, 1)
    binder_replicates_df = replicate_df.copy()[replicate_df['name'].apply(tuple,1).isin(binder_idx)]

    fig = plt.figure(figsize=(7.5,9))
    plt.subplots_adjust(wspace=0.1)

    # plot 1
    df = binder_replicates_df[['fit_Kd','name']].groupby('name').agg('mean')
    df.sort_values('fit_Kd', ascending=False, inplace=True)
    df.reset_index(inplace=True)

    # Draw horizontal lines
    ax = plt.subplot(2,2,1)
    ax.set_title('Average Affinity < 1',size=10)
    ax.hlines(y=df.index, xmin=0, xmax=10, color='gray', alpha=0.5, linewidth=.1, linestyles='-')

    # Draw the Dots
    for i, name in enumerate(df.name):
        df_Kd = replicate_df.loc[replicate_df.name==name, :]
        ax.scatter(y=np.repeat(i, df_Kd.shape[0]), x='fit_Kd', data=df_Kd, s=5, edgecolors='dodgerblue', c='w', alpha=0.5)
        #ax.scatter(y=i, x='fit_Kd', data=df.loc[df.name==name, :], s=5, edgecolors='firebrick', c='w', alpha=0.5)
    ax.set_xlim([0,10])
    ax.set_ylabel('Each row contains all replicates\nfor a single Domain-Peptide Pair\n(rows ordered by increasing mean affinity)',size=9)
    ax.set_yticks([''])
    ax.set_yticklabels([''])
    ax.tick_params(axis='both', which='major', labelsize=9)

    # plot 2
    binder_df = grouped_df[(grouped_df.functional_binding_call=='binder') & (grouped_df.count_bind>1) & (grouped_df.fit_Kd>1) & (grouped_df.fit_Kd<=3)]
    binder_idx = binder_df['name'].drop_duplicates().reset_index(drop=True).apply(tuple, 1)
    binder_replicates_df = replicate_df.copy()[replicate_df['name'].apply(tuple,1).isin(binder_idx)]

    df = binder_replicates_df[['fit_Kd','name']].groupby('name').agg('mean')
    df.sort_values('fit_Kd', ascending=False, inplace=True)
    df.reset_index(inplace=True)

    # Draw horizontal lines
    ax = plt.subplot(2,2,2)
    ax.set_title('1 $\leq$ Average Affinity < 3',size=10)
    ax.hlines(y=df.index, xmin=0, xmax=20, color='gray', alpha=0.5, linewidth=.1, linestyles='-')


    # Draw the Dots
    for i, name in enumerate(df.name):
        df_Kd = replicate_df.loc[replicate_df.name==name, :]
        ax.scatter(y=np.repeat(i, df_Kd.shape[0]), x='fit_Kd', data=df_Kd, s=5, edgecolors='dodgerblue', c='w', alpha=0.5)
        #ax.scatter(y=i, x='fit_Kd', data=df.loc[df.name==name, :], s=5, edgecolors='firebrick', c='w', alpha=0.5)
    ax.set_xlim([0, 20])
    ax.set_yticks([''])
    ax.set_yticklabels([''])
    ax.tick_params(axis='both', which='major', labelsize=9)

    # plot 3
    binder_df = grouped_df[(grouped_df.functional_binding_call=='binder') & (grouped_df.count_bind>1) & (grouped_df.fit_Kd>3) & (grouped_df.fit_Kd<=6)]
    binder_idx = binder_df['name'].drop_duplicates().reset_index(drop=True).apply(tuple, 1)
    binder_replicates_df = replicate_df.copy()[replicate_df['name'].apply(tuple,1).isin(binder_idx)]

    df = binder_replicates_df[['fit_Kd','name']].groupby('name').agg('mean')
    df.sort_values('fit_Kd', ascending=False, inplace=True)
    df.reset_index(inplace=True)

    # Draw horizontal lines
    ax = plt.subplot(2,2,3)
    ax.set_title('3 $\leq$ Average Affinity < 6',size=10)
    ax.hlines(y=df.index, xmin=0, xmax=30, color='gray', alpha=0.5, linewidth=.1, linestyles='-')

    # Draw the Dots
    for i, name in enumerate(df.name):
        df_Kd = replicate_df.loc[replicate_df.name==name, :]
        ax.scatter(y=np.repeat(i, df_Kd.shape[0]), x='fit_Kd', data=df_Kd, s=5, edgecolors='dodgerblue', c='w', alpha=0.5)
        #ax.scatter(y=i, x='fit_Kd', data=df.loc[df.name==name, :], s=5, edgecolors='firebrick', c='w', alpha=0.5)
    ax.set_xlim([0, 30])
    ax.set_ylabel('Each row contains all replicates\nfor a single Domain-Peptide Pair\n(rows ordered by increasing mean affinity)',size=9)
    ax.set_xlabel('Replicate Affinity ($\mu$M)',size=9)
    ax.set_yticks([''])
    ax.set_yticklabels([''])
    ax.tick_params(axis='both', which='major', labelsize=9)


    # plot 4
    binder_df = grouped_df[(grouped_df.functional_binding_call=='binder') & (grouped_df.count_bind>1) & (grouped_df.fit_Kd>6)]
    binder_idx = binder_df['name'].drop_duplicates().reset_index(drop=True).apply(tuple, 1)
    binder_replicates_df = replicate_df.copy()[replicate_df['name'].apply(tuple,1).isin(binder_idx)]

    df = binder_replicates_df[['fit_Kd','name']].groupby('name').agg('mean')
    df.sort_values('fit_Kd', ascending=False, inplace=True)
    df.reset_index(inplace=True)

    # Draw horizontal lines
    ax = plt.subplot(2,2,4)
    ax.set_title('6 $\leq$ Average Affinity',size=10)
    ax.hlines(y=df.index, xmin=0, xmax=40, color='gray', alpha=0.5, linewidth=.1, linestyles='-')

    # Draw the Dots
    for i, name in enumerate(df.name):
        df_Kd = replicate_df.loc[replicate_df.name==name, :]
        ax.scatter(y=np.repeat(i, df_Kd.shape[0]), x='fit_Kd', data=df_Kd, s=5, edgecolors='dodgerblue', c='w', alpha=0.5)
        #ax.scatter(y=i, x='fit_Kd', data=df.loc[df.name==name, :], s=5, edgecolors='firebrick', c='w', alpha=0.5)
    ax.set_xlim([0, 40])
    ax.set_xlabel('Replicate Affinity ($\mu$M)',size=9)
    ax.set_yticks([''])
    ax.set_yticklabels([''])
    ax.tick_params(axis='both', which='major', labelsize=9)

    fig.savefig(output_filename, bbox_inches='tight', dpi=1200)


def report_replicate_tracking(rep_df,text_descriptor,binding_field,figs_wide=12,figs_tall=7,xticks_on=True):
    # binding_field = 'binding_call' or 'binding_call_functionality_evaluated'

    fig=plt.figure(figsize=(6.5,8))
    fig.subplots_adjust(hspace=0.4)
    fig_idx = 1
    for domain, domain_df in rep_df.reset_index().groupby(['domain']):
        if domain:

            plot_df = pf.imperfect_merge(domain_df, domain, binding_field)

            # create plottable df
            binding_call_cols = [col for col in plot_df.columns if 'binding_call' in col]
            plot_df = plot_df[binding_call_cols]

            # 1000 is not measured
            # if binding_field == 'binding_call':
            #     plot_df = plot_df.applymap(lambda x: 0 if 'nobinding' in str(x) else 1 if 'binder' in str(x) else 2 if 'aggregator' in str(x) else 3 if 'low-SNR' in str(x) else 1000)
            # if binding_field == 'binding_call_functionality_evaluated':
            #     plot_df = plot_df.applymap(lambda x: 0 if 'nobinding' in str(x) else 1 if 'binder' in str(x) else 2 if 'aggregator' in str(x) else 3 if 'low-SNR' in str(x) else 4 if 'non-functional' in str(x) else 1000)
            if binding_field == 'binding_call':
                plot_df = plot_df.applymap(lambda x: 0 if 'nobinding' in str(x) else 1 if 'binder' in str(x) else 1000)
            if binding_field == 'binding_call_functionality_evaluated':
                plot_df = plot_df.applymap(lambda x: 0 if 'nobinding' in str(x) else 1 if 'binder' in str(x) else 4 if 'non-functional' in str(x) else 1000)

            ax = plt.subplot(figs_tall,figs_wide,fig_idx)
            pf.subplot_heatmap_colormesh(plot_df,domain,ax,xticks_on)
            fig_idx += 1

    # ax = plt.subplot(figs_tall, figs_wide+1, figs_tall*(figs_wide+1))
    # ax.axis('off')
    # patch1 = mpatches.Patch(facecolor='white', edgecolor = 'k', linewidth =1, label='Non-Binder')
    # patch2 = mpatches.Patch(color='green', label='Binder')
    # patch5 = mpatches.Patch(color='dodgerblue', label='Non-Functional')
    # patch6= mpatches.Patch(color='lightgray', label='Not Measured')
    # plt.legend(handles=[patch1,patch2,patch5,patch6], loc = 'lower right',fontsize=6)

    fig.savefig('figures/'+text_descriptor+'.pdf',bbox_inches='tight')
    #fig.savefig('Dissertation-JFPRepTrk '+text_descriptor+'.eps',bbox_inches='tight')
    plt.close('all')


def report_NonFunctionalProtein_Examples(df, domain_list,text_descriptor):
    fig = plt.figure(figsize=(4.5, 4))
    fig.subplots_adjust(hspace=0.4)
    figs_wide = 3
    figs_tall = 3

    for i, domain in enumerate(domain_list, start=1):
        temp_df = df.copy()
        temp_df = temp_df[temp_df.index.get_level_values('domain').isin([domain])]
        ax = plt.subplot(figs_tall, figs_wide, 3 * i - 2)
        if i == 3:
            pf.plot_domain_results(ax, temp_df, 'binding_call', ylabel_on=True, xlabel_on=True)
        else:
            pf.plot_domain_results(ax, temp_df, 'binding_call', ylabel_on=True)

        ax = plt.subplot(figs_tall, figs_wide, 3 * i -1)
        if i == 3:
            pf.plot_domain_results(ax, temp_df, 'binding_call_functionality_evaluated', xlabel_on=True)
        else:
            pf.plot_domain_results(ax, temp_df, 'binding_call_functionality_evaluated')

    ax = plt.subplot(figs_tall, figs_wide, figs_tall*figs_wide)
    ax.axis('off')
    patch1 = mpatches.Patch(facecolor='white', edgecolor = 'k', linewidth =1, label='Non-Binder')
    patch2 = mpatches.Patch(color='green', label='Binder')
    #patch3 = mpatches.Patch(color='darkorange', label='Aggregator')
    #patch4 = mpatches.Patch(color='dimgray', label='low-SNR')
    patch5 = mpatches.Patch(color='dodgerblue', label='Non-Functional')
    patch6= mpatches.Patch(color='lightgray', label='Not Measured')

    plt.legend(handles=[patch1,patch2,patch5,patch6], loc = 'lower right',fontsize=6)

    fig.savefig(text_descriptor+'.pdf', bbox_inches='tight')
    plt.close('all')


def report_CorrelationScatter1x3(filename, datasets, autolabels=True):
    fig = plt.figure(figsize=(7.2, 2.6))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5)

    fig_qty = len(datasets) - 1
    fig_idx = pf.get_upper_triangle_index(fig_qty)

    for i, (data1, data2) in enumerate(it.combinations(datasets, 2)):
        #ax = plt.subplot(fig_qty, fig_qty, fig_idx[i])
        ax = plt.subplot(1, 3, i+1)
        dataname1 = data1[0]
        dataset1 = data1[1]
        dataname2 = data2[0]
        dataset2 = data2[1]
        result_df = pd.merge(dataset1, dataset2, how='inner', left_on=['gene_name', 'pY_pos', 'domain'],
                             right_on=['gene_name', 'pY_pos', 'domain'])
        print dataname1, dataname2, result_df.gene_name.unique()
        result_df['Kd_x'] = result_df['Kd_x'].astype(float)
        result_df['Kd_y'] = result_df['Kd_y'].astype(float)
        if result_df.empty:
            plt.scatter(0, 0, s=0, color='DarkBlue', edgecolors='none')

        else:
            result_df.plot.scatter(x='Kd_y', y='Kd_x', ax=ax, s=3, color='DarkBlue', edgecolors='none', fontsize=6)
            r = stats.pearsonr(result_df['Kd_y'].values, result_df['Kd_x'].values)
            ax.text(0.65, 0.9, 'r = ' + str.format('{:.3f}', r[0]), size=6, transform=ax.transAxes)
            ax.text(0.65, 0.85, 'n = ' + str.format('{0:d}', len(result_df['Kd_x'].values)), size=6, transform=ax.transAxes)

        if dataname1 in ['JonesFP', 'Jones2012', 'Jones2014', 'Jones 2012-14']:
            ax.set_ylim([-0.5, 21])
            ax.set_yticks([0, 20])
            ax.set_yticklabels(['0', '10','20'], size=6)
        else:
            ax.set_ylim([-0.05, 2.1])
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['0', '1', '2'], size=6)

        if dataname2 in ['JonesFP', 'Jones2012', 'Jones2014', 'Jones 2012-14']:
            ax.set_xlim([-0.5, 21])
            ax.set_xticks([0, 10, 20])
            ax.set_xticklabels(['0', '10','20'], size=6)
        else:
            ax.set_xlim([-0.05, 2.1])
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['0', '1', '2'], size=6)

        # if dataname1 or dataname2 in ['JonesFP', 'Jones2012', 'Jones2014', 'Jones 2012-14']:
        #     ax.set_ylim([-0.5, 21])
        #     ax.set_yticks([0, 20])
        #     ax.set_yticklabels(['0', '20'], size=6)
        #     ax.set_xlim([-0.5, 21])
        #     ax.set_xticks([0, 20])
        #     ax.set_xticklabels(['0', '20'], size=6)
        # else:
        #     ax.set_ylim([-0.05, 2.1])
        #     ax.set_yticks([0, 2])
        #     ax.set_yticklabels(['0', '2'], size=6)
        #     ax.set_xlim([-0.05, 2.1])
        #     ax.set_xticks([0, 2])
        #     ax.set_xticklabels(['0', '2'], size=6)


        if autolabels == True:
            if fig_idx[i] in list(1 + np.arange(fig_qty)):
                ax.set_title(dataname2, size=6)
                ax.set_xlabel('')
                ax.set_ylabel('')
                if fig_idx[i] == 1:
                    ax.set_ylabel(dataname1, size=6)
            elif fig_idx[i] in [(1 * fig_qty) + 2, (2 * fig_qty) + 3, (3 * fig_qty) + 4, (4 * fig_qty) + 5,
                                (5 * fig_qty) + 6, ]:
                ax.set_ylabel('$\mu$M\n'+dataname1, size=6)
                ax.set_xlabel('')
            else:
                ax.set_xlabel('')
                ax.set_ylabel('')
        else:
            #ax.set_title(dataname2, size=6)
            ax.set_ylabel(dataname1+' ($\mu$M)', size=8)
            ax.set_xlabel(dataname2+' ($\mu$M)', size=8)

        x = np.linspace(0, 20, 500)
        ax.plot(x, x, '--', linewidth=1, color='lightgray')
    plt.tight_layout()
    plt.savefig('figures/'+filename + '-CorrelationScatter.pdf', bbox_inches='tight')
    #plt.savefig('figures/'+filename + '-CorrelationScatter.svg', bbox_inches='tight', pad_inches=0.1)
    plt.close()


def report_CorrelationScatter(filename, datasets, autolabels=True):
    fig = plt.figure(figsize=(11, 8.5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5)

    fig_qty = len(datasets) - 1
    fig_idx = pf.get_upper_triangle_index(fig_qty)

    for i, (data1, data2) in enumerate(it.combinations(datasets, 2)):
        ax = plt.subplot(fig_qty, fig_qty, fig_idx[i])
        dataname1 = data1[0]
        dataset1 = data1[1]
        dataname2 = data2[0]
        dataset2 = data2[1]
        result_df = pd.merge(dataset1, dataset2, how='inner', left_on=['gene_name', 'pY_pos', 'domain'],
                             right_on=['gene_name', 'pY_pos', 'domain'])
        result_df['Kd_x'] = result_df['Kd_x'].astype(float)
        result_df['Kd_y'] = result_df['Kd_y'].astype(float)
        if result_df.empty:
            plt.scatter(0, 0, s=0, color='DarkBlue', edgecolors='none')

        else:
            result_df.plot.scatter(x='Kd_y', y='Kd_x', ax=ax, s=3, color='DarkBlue', edgecolors='none', fontsize=6)
            r = stats.pearsonr(result_df['Kd_y'].values, result_df['Kd_x'].values)
            ax.text(0.65, 0.9, 'r = ' + str.format('{:.3f}', r[0]), size=5, transform=ax.transAxes)

        if dataname1 in ['JonesFP', 'Jones2012', 'Jones2014', 'Jones 2012-14']:
            ax.set_ylim([-0.5, 21])
            ax.set_yticks([0, 20])
            ax.set_yticklabels(['0', '10','20'], size=6)
        else:
            ax.set_ylim([-0.05, 2.1])
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['0', '1', '2'], size=6)

        if dataname2 in ['JonesFP', 'Jones2012', 'Jones2014', 'Jones 2012-14']:
            ax.set_xlim([-0.5, 21])
            ax.set_xticks([0, 10, 20])
            ax.set_xticklabels(['0', '10','20'], size=6)
        else:
            ax.set_xlim([-0.05, 2.1])
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['0', '1', '2'], size=6)

        if dataname1 or dataname2 in ['JonesFP', 'Jones2012', 'Jones2014', 'Jones 2012-14']:
            ax.set_ylim([-0.5, 21])
            ax.set_yticks([0, 20])
            ax.set_yticklabels(['0', '20'], size=6)
            ax.set_xlim([-0.5, 21])
            ax.set_xticks([0, 20])
            ax.set_xticklabels(['0', '20'], size=6)
        else:
            ax.set_ylim([-0.05, 2.1])
            ax.set_yticks([0, 2])
            ax.set_yticklabels(['0', '2'], size=6)
            ax.set_xlim([-0.05, 2.1])
            ax.set_xticks([0, 2])
            ax.set_xticklabels(['0', '2'], size=6)


        if autolabels == True:
            if fig_idx[i] in list(1 + np.arange(fig_qty)):
                ax.set_title(dataname2, size=6)
                ax.set_xlabel('')
                ax.set_ylabel('')
                if fig_idx[i] == 1:
                    ax.set_ylabel(dataname1, size=6)
            elif fig_idx[i] in [(1 * fig_qty) + 2, (2 * fig_qty) + 3, (3 * fig_qty) + 4, (4 * fig_qty) + 5,
                                (5 * fig_qty) + 6, ]:
                ax.set_ylabel('$\mu$M\n'+dataname1, size=6)
                ax.set_xlabel('')
            else:
                ax.set_xlabel('')
                ax.set_ylabel('')
        else:
            #ax.set_title(dataname2, size=6)
            ax.set_ylabel(dataname1+' ($\mu$M)', size=6)
            ax.set_xlabel(dataname2+' ($\mu$M)', size=6)

        x = np.linspace(0, 20, 500)
        ax.plot(x, x, '--', linewidth=1, color='lightgray')
    plt.tight_layout()
    plt.savefig('figures/'+filename + '-CorrelationScatter.pdf')
    #plt.savefig('figures/'+filename + '-CorrelationScatter.svg', bbox_inches='tight', pad_inches=0.1)
    plt.close()


def report_Kdexampleplots(print_data, output_file_name):
    rows_per_page = 4
    columns_per_page = 3

    fig = plt.figure(figsize=(6.5, 6.5))
    plt.subplots_adjust(hspace=0.3, wspace=0.8)
    figure_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    for i, row in enumerate(print_data.itertuples()):

        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
        ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                              (plot_location['gridpos_y'], plot_location['gridpos_x']),
                              rowspan=plot_location['rowspan'], colspan=plot_location['colspan'])

        ax.axis('off')
        #if row.fit_Kd < 1:
        #    kd_representation = round(row.fit_Kd, 1)
        #else:
        #    kd_representation = math.floor(row.fit_Kd)
        #ax.text(0, .9, figure_labels[i], size=12)
        #ax.text(0, .7, r'$K_d \approx %2.1f$' % kd_representation, size=12)
        #ax.text(0, .5, fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'], size=8)
        # ax.text(0,0,fig_data['peptide']+' pY'+fig_data['pY_pos'],size=12)
        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='Kd_example',ylabel = fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)',title='')
        elif i == rows_per_page-1:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='Kd_example',ylabel = fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='Kd_example',ylabel = fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)')

        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='Kd_example',title='')
        elif i == rows_per_page-1:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='Kd_example',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='Kd_example')

    plt.tight_layout()
    fig.savefig(output_file_name+'.pdf',bbox_inches='tight')
    fig.clf()
    plt.close()


def report_varianceexampleplots(print_data, output_file_name):
    rows_per_page = 3
    columns_per_page = 3

    fig = plt.figure(figsize=(6.5, 5))
    plt.subplots_adjust(hspace=0.3, wspace=0.8)

    for i, row in enumerate(print_data.itertuples()):
        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i // columns_per_page, i % columns_per_page)
        pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='variance', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'] + '\n' + 'FP (mP)', title='', Kd_fixedlabel=True)


    plt.tight_layout()
    fig.savefig(output_file_name + '.pdf', bbox_inches='tight')
    fig.clf()
    plt.close()


def report_BGexampleplots(print_data, output_file_name):
    rows_per_page = 4
    columns_per_page = 2

    fig = plt.figure(figsize=(6,9.5))
    plt.subplots_adjust(hspace=0.3,wspace=0.5)
    #figure_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    #figure_descriptions = ['Low Background','High Background','Expected Background']
    for i, row in enumerate(print_data.itertuples()):

        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
        ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                              (plot_location['gridpos_y'], plot_location['gridpos_x']),
                              rowspan=plot_location['rowspan'], colspan=plot_location['colspan'])
        ax.axis('off')
        if row.fit_Kd < 1:
            kd_representation = round(row.fit_Kd, 1)
        else:
            kd_representation = math.floor(row.fit_Kd)
        #ax.text(0, .9, figure_labels[i], size=12)
        #ax.text(0, .7, figure_descriptions[i], size=10)
        #ax.text(0, .5, fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'], size=8)

        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='bg', ylabel = fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)', title = '')
        elif i == rows_per_page-2:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='bg', ylabel = fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='bg', ylabel = fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)')


        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='bg', title = '')
        elif i == rows_per_page-2:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='bg',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='bg')

        if i == 2:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, 3,1)
            ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                                  (plot_location['gridpos_y'], plot_location['gridpos_x']),
                                  rowspan=plot_location['rowspan'],
                                  colspan=plot_location['colspan'])
            ax.plot(0, 0, linestyle=':', color='gray', label='Fit offset', linewidth=1)
            ax.plot(0, 0, linestyle=':', color='orange', label='Published BG', linewidth=1)
            ax.plot(0, 0, color='b', label='One-to-one', linewidth=0.5, linestyle='--')
            ax.axis('off')
            ax.legend(loc='upper left', fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(output_file_name+'.pdf')
    fig.clf()
    plt.close()


def report_SNRexampleplots(print_data, output_file_name):
    rows_per_page = 5
    columns_per_page = 3

    fig = plt.figure(figsize=(6.5, 7.5))
    plt.subplots_adjust(hspace=0.3,wspace=0.5)
    figure_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    figure_descriptions = ['SNR ~ 3.0','SNR ~ 2.0','SNR ~ 1.0','Low-SNR (outlier)', 'Low-SNR (noise)']
    for i, row in enumerate(print_data.itertuples()):

        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
        ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                              (plot_location['gridpos_y'], plot_location['gridpos_x']),
                              rowspan=plot_location['rowspan'], colspan=plot_location['colspan'])
        ax.axis('off')
        if row.fit_Kd < 1:
            kd_representation = round(row.fit_Kd, 1)
        else:
            kd_representation = math.floor(row.fit_Kd)
        #ax.text(0, .9, figure_labels[i], size=12)
        ax.text(0, .7, figure_descriptions[i], size=10)
        #ax.text(0, .5, fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'], size=8)
        # ax.text(0,0,fig_data['peptide']+' pY'+fig_data['pY_pos'],size=12)
        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='SNR', title = '', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)')
        elif i == rows_per_page-1:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='SNR', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='SNR', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)')


        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='SNR', title = '')
        elif i == rows_per_page-1:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='SNR',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='SNR')


    plt.tight_layout()
    fig.savefig(output_file_name+'.pdf',bbox_inches='tight')
    fig.clf()
    plt.close()


def report_NonBinderexampleplots(print_data, output_file_name):
    rows_per_page = 3
    columns_per_page = 2

    fig = plt.figure(figsize=(4.5, 5.75 ))
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    figure_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']

    for i, row in enumerate(print_data.itertuples()):

        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
        ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                              (plot_location['gridpos_y'], plot_location['gridpos_x']),
                              rowspan=plot_location['rowspan'], colspan=plot_location['colspan'])
        ax.axis('off')
        if row.fit_Kd < 1:
            kd_representation = round(row.fit_Kd, 1)
        else:
            kd_representation = math.floor(row.fit_Kd)


        #ax.text(0, .9, figure_labels[i], size=12)
        #ax.text(0, .7, fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'], size=6)

        # Figure
        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, type='binding_ylimon', title = 'Full Scale', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)')
        elif i == rows_per_page - 1:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
            pf.plot_Kd_curves_examples_v2(['fo', 'linear'], fig_data, yfits, plot_location, type='binding_ylimon', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
            pf.plot_Kd_curves_examples_v2(['fo', 'linear'], fig_data, yfits, plot_location, type='binding_ylimon', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos']+ '\n'+ 'FP (mP)')


        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, type='binding_zoomed', title = 'Zoomed')
        elif i == rows_per_page -1:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, type='binding_zoomed',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, type='binding_zoomed')

        # Figure
        #if i == 0:
        #    plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
        #    pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, logplots=True, type='binding', title = 'Semi-Log')

        #else:
        #    plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
        #    pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, logplots=True, type='binding')

    plt.tight_layout()
    fig.savefig(output_file_name+'.pdf')
    fig.clf()
    plt.close()


def report_ModelSelectionexampleplots(print_data, output_file_name):
    rows_per_page = 4
    columns_per_page = 3

    fig = plt.figure(figsize=(6.5, 7.5))
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    figure_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    figure_descriptions = ['Fit Selected: one-to-one\nK$_d$ ~3$\mu$M', 'Fit Selected: one-to-one\nK$_d$ ~7$\mu$M','Fit Selected: one-to-one\nK$_d$ ~15$\mu$M','Fit Selected: linear\nNon-Binder']
    for i, row in enumerate(print_data.itertuples()):

        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
        ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                              (plot_location['gridpos_y'], plot_location['gridpos_x']),
                              rowspan=plot_location['rowspan'], colspan=plot_location['colspan'])
        ax.axis('off')
        if row.fit_Kd < 1:
            kd_representation = round(row.fit_Kd, 1)
        else:
            kd_representation = math.floor(row.fit_Kd)
        #ax.text(0, .9, figure_labels[i], size=12)
        ax.text(0, .7, 'linear AICc = '+str(round(fig_data['linear_AICc'],1)), size=8)
        ax.text(0, .55, 'one-to-one AICc = '+str(round(fig_data['fo_AICc'],1)), size=8)
        ax.text(0, .3, figure_descriptions[i], size=8)
        #ax.text(0, .1, fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'], size=8)


        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, type='model_selection', title = '', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'])
        elif i == rows_per_page-1 :
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, type='model_selection', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'],xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, type='model_selection', ylabel=fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'])


        # Figure
        if i == 0:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, logplots=True, type='model_selection', title = '')
        elif i == rows_per_page - 1:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, logplots=True, type='model_selection',xlabel=True)
        else:
            plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
            pf.plot_Kd_curves_examples_v2(['fo','linear'], fig_data, yfits, plot_location, logplots=True, type='model_selection')


    plt.tight_layout()
    fig.savefig(output_file_name+'.pdf')
    fig.clf()
    plt.close()


def report_Saturationexampleplots(print_data, output_file_name):
    rows_per_page = 5
    columns_per_page = 3

    fig = plt.figure(figsize=(6.5, 7.5))
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    figure_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    figure_descriptions = ['', '','','','']
    for i, row in enumerate(print_data.itertuples()):

        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
        ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                              (plot_location['gridpos_y'], plot_location['gridpos_x']),
                              rowspan=plot_location['rowspan'], colspan=plot_location['colspan'])
        ax.axis('off')
        if row.fit_Kd < 1:
            kd_representation = round(row.fit_Kd, 1)
        else:
            kd_representation = math.floor(row.fit_Kd)
        ax.text(0, .9, figure_labels[i], size=12)
        ax.text(0, .7, 'Saturation ($F_{max}$-$F_{0}$) = '+str(round(fig_data['fo_Fmax']-fig_data['fo_offset'],1)), size=8)
        ax.text(0, .55, '$K_d$ = ' + str(round(fig_data['fo_fitKd'], 2) +'$\mu$M'),size=8)
        ax.text(0, .3, fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'], size=8)
        #ax.text(0, .3, figure_descriptions[i], size=8)


        # Figure
        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
        pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, type='SNR')

        # Figure
        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
        pf.plot_Kd_curves_examples_v2(['fo'], fig_data, yfits, plot_location, logplots=True, type='SNR')

    fig.savefig(output_file_name+'.pdf',bbox_inches='tight')
    fig.clf()
    plt.close()


def report_exampleplots(print_data, output_file_name):
    rows_per_page = 8
    columns_per_page = 3

    fig = plt.figure(figsize=(6.5, 9))
    plt.subplots_adjust(hspace=0.3,wspace=0.5)
    figure_labels = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    figure_descriptions = ['Binder', 'Non-Binder','Aggregator','Low-SNR']
    for i, row in enumerate(print_data.itertuples()):

        fit_parameters = pf.set_fit_parameters(row)
        fig_data = pf.get_fig_data(row)
        yfits = pf.generate_yfits(fig_data, fit_parameters)

        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 0)
        ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                              (plot_location['gridpos_y'], plot_location['gridpos_x']),
                              rowspan=plot_location['rowspan'], colspan=plot_location['colspan'])
        ax.axis('off')
        if row.fit_Kd < 1:
            kd_representation = round(row.fit_Kd, 1)
        else:
            kd_representation = math.floor(row.fit_Kd)
        ax.text(0, .9, figure_labels[i], size=12)
        ax.text(0, .7, figure_descriptions[i], size=10)
        ax.text(0, .5, fig_data['domain'] + ', ' + fig_data['peptide'] + ' pY' + fig_data['pY_pos'], size=8)
        # ax.text(0,0,fig_data['peptide']+' pY'+fig_data['pY_pos'],size=12)
        # Figure
        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 1)
        pf.plot_Kd_curves_examples(['fo'], fig_data, yfits, plot_location, ylim='on', offset='off', bg_disp='off',
                                   fmax='off', kd='off')

        # Figure
        plot_location = pf.set_plot_location(rows_per_page, columns_per_page, i % rows_per_page, 2)
        pf.plot_Kd_curves_examples(['fo'], fig_data, yfits, plot_location, logplots=True, ylim='on', offset='off',
                                   bg_disp='off', fmax='off', kd='off')

    fig.savefig(output_file_name+'.pdf',bbox_inches='tight')
    fig.clf()
    plt.close()


def report_theoreticalplots(output_file_name):
    rows_per_page = 1
    columns_per_page = 2

    fig = plt.figure(figsize=(6.5, 3))
    plt.subplots_adjust(wspace=0.4)

    plot_location = pf.set_plot_location(rows_per_page, columns_per_page,  0, 0)
    pf.theoretical_Kd_plot(plot_location)
    plot_location = pf.set_plot_location(rows_per_page, columns_per_page, 0, 1)
    pf.theoretical_Kd_plot(plot_location, logplot=True)

    plt.tight_layout()
    fig.savefig(output_file_name+'.pdf')
    fig.clf()
    plt.close()


def report_degradation_theoretical(output_file_name):
    ## Plot of effect of degredation on actual Kd vs measured Kd

    rows_per_page = 1
    columns_per_page = 1

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(hspace=0.3, wspace=0.5)

    plot_location = pf.set_plot_location(rows_per_page, columns_per_page, 0, 0)
    Kd = 4
    Fmax = 100
    offset = 0

    ax = plt.subplot2grid((plot_location['gridsize_y'], plot_location['gridsize_x']),
                          (plot_location['gridpos_y'], plot_location['gridpos_x']), rowspan=plot_location['rowspan'],
                          colspan=plot_location['colspan'])
    conc = np.append(np.linspace(0.001, 0.1, 1000),
                     np.linspace(0.2, 10, 1000))
    conc = np.append(conc, np.linspace(11, 1100, 1000))

    y_vals = cff.model_firstorder_offset(conc, Fmax + offset, Kd, offset)

    datapoint_conc = np.asarray([10.0 / (2 ** x) for x in np.arange(0, 12)][::-1])
    datapoint_y_vals = cff.model_firstorder_offset(datapoint_conc, Fmax + offset, Kd, offset)

    ax.set_title('Errors in Protein Concentration Affect\nDerived $K_d$ Parameter', size=8)

    ax.plot(datapoint_conc, datapoint_y_vals, marker='o', linestyle='', markeredgecolor='k', markerfacecolor='white')
    ax.plot(conc, y_vals, color='k', linestyle=':')

    ax.set_xlim([-0.1, 10.1])
    ax.set_ylim([0.9 * offset, 1.1 * (offset + Fmax)])
    ax.set_yticks([Fmax / 2, Fmax])
    ax.set_yticklabels(['$\\frac{1}{2}$$F$$_{max}$', '$F$$_{max}$'], size=6)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xticklabels(['0', '2', '4', '6', '8', '10'], size=6)

    ax.axhline(Fmax, color='dodgerblue', linestyle='--', linewidth=1)
    ax.axhline(Fmax / 2, color='dodgerblue', linestyle='--', linewidth=1)
    ax.plot([Kd, Kd], [0, Fmax / 2], color='firebrick', linestyle='--', linewidth=1)

    plt.text(Kd - .12, -14, '$K_d$ =  ' + str(Kd * 1.0) + '$\mu$$M$', color='k', size=6, rotation=90)

    conc_modifier_list = [0.75, 0.50, 0.25]
    color_list = ['dimgray', 'darkgray', 'lightgray']

    for conc_mod, clr in zip(conc_modifier_list, color_list):
        ax.plot(datapoint_conc * conc_mod, datapoint_y_vals, marker='o', linestyle='', markeredgecolor=clr,
                markerfacecolor='white')
        ax.plot(conc * conc_mod, y_vals, color=clr, linestyle=':')
        ax.plot([Kd * conc_mod, Kd * conc_mod], [0, Fmax / 2], color='firebrick', linestyle='--', linewidth=1)
        plt.text(Kd * conc_mod - .12, -14, '$K_d$ =  ' + str(Kd * conc_mod) + '$\mu$$M$', color=color_list[0], size=6,
                 rotation=90)

    custom_legend_entries = ['100% active protein'] + [str(x * 100) + '% active protein' for x in conc_modifier_list]
    custom_lines = [plt.Line2D([0], [0], color='k', lw=1, ls=':'),
                    plt.Line2D([0], [0], color=color_list[0], lw=1, ls=':'),
                    plt.Line2D([0], [0], color=color_list[1], lw=1, ls=':'),
                    plt.Line2D([0], [0], color=color_list[2], lw=1, ls=':')]

    ax.legend(custom_lines, custom_legend_entries, fontsize=6, loc='lower right')
    fig.savefig(output_file_name, bbox_inches='tight')
    fig.clf()
    plt.close()


def report_histogram_SNR(df,output_file_name):
    ## Plot of effect of degredation on actual Kd vs measured Kd
    df = df[(df['binding_call']=='binder')| (df['binding_call']=='low-SNR')].sort_values('fit_snr',ascending=False).copy()
    num_binders = df[(df['binding_call']=='binder')].shape[0]
    num_lowSNR = df[(df['binding_call']=='low-SNR')].shape[0]
    perc_lowSNR = round(100* num_lowSNR / (num_lowSNR+num_binders),1)
    perc_bind = round(100* num_binders / (num_lowSNR+num_binders),1)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    ax.hist(df.fit_snr.values, normed=True, bins=100)
    ax.axvline(1, color='r', linestyle='--', linewidth=1)
    ax.text(1.3,0.8, 'SNR = 1.0', color='k', size=8,rotation=90)
    ax.text(3,0.8, str(num_lowSNR)+' binders with SNR < 1.0 ('+str(perc_lowSNR)+'%)', color='k', size=8)
    ax.text(3,0.7, str(num_binders)+' binders with SNR <= 1.0 ('+str(perc_bind)+'%)', color='k', size=8)
    plt.tight_layout()
    fig.savefig(output_file_name)
    fig.clf()
    plt.close()


def report_scatter_rsq_vs_snr(print_df,output_filename):

    print_df = print_df[['fit_rsq', 'fit_snr']]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    xdata = print_df['fit_rsq'].values
    ydata = print_df['fit_snr'].values

    plt.scatter(xdata, ydata, marker='o', s=10, facecolor='none', edgecolor='b', linewidths=0.5, rasterized=True)
    ax.set_xlabel(r'Fit $r^2$')
    ax.set_ylabel(r'Fir Signal-to-Noise (SNR) Metric')
    ax.set_title('R-squared vs Signal-To Noise Metrics for Binders and Low-SNR Fits')

    ax.axhline(1, color='r', linestyle='--', linewidth=1)
    ax.text(0.1,1.1, 'SNR = 1.0', color='r', size=8)

    ax.axvline(0.95, color='r', linestyle='--', linewidth=1)
    ax.text(0.9,5, '$r^2$ $\geq$ 0.95', color='r', size=8,rotation=90)

    fig.savefig(output_filename, bbox_inches='tight', dpi=600)
    plt.close('all')


def report_hist_Fmax(df,output_filename):

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)

    print_df = df.copy()[(df.binding_call_functionality_evaluated == 'binder') & (df.fit_Kd <= 20) & (df.fit_snr > 1)].sort_values('fit_Kd',ascending=False)
    series01_count = str(print_df.shape[0])
    ax.hist(print_df.fit_Fmax,bins=100,normed=False,color = 'navy')

    print_df = df.copy()[(df.binding_call_functionality_evaluated == 'binder') & (df.fit_Kd <= 10) & (df.fit_snr > 1)]
    series02_count = str(print_df.shape[0])
    ax.hist(print_df.fit_Fmax,bins=100,color='blue',normed=False)

    print_df = df.copy()[(df.binding_call_functionality_evaluated == 'binder') & (df.fit_Kd <= 5) & (df.fit_snr > 1)]
    series03_count = str(print_df.shape[0])
    ax.hist(print_df.fit_Fmax,bins=100,color='royalblue',normed=False)

    print_df = df.copy()[(df.binding_call_functionality_evaluated == 'binder') & (df.fit_Kd <= 1) & (df.fit_snr > 1)]
    series04_count = str(print_df.shape[0])
    ax.hist(print_df.fit_Fmax,bins=100,color='dodgerblue',normed=False)

    ax.legend(['Kd<20 ('+series01_count+')','Kd<10 ('+series02_count+')','Kd<5 ('+series03_count+')','Kd<1 ('+series04_count+')'])
    ax.set_xlabel(r'Fluorescence Saturation ($F_{max}$)')

    fig.savefig(output_filename, bbox_inches='tight', dpi=600)
    plt.close('all')


import time
import pandas as pd
from ms_func_lib import rawdata_load_functions as rdlf
from ms_func_lib import ms_utils


def identify_replicate_groups_with_one_or_more_binders(df):
    # Find any binder in the domain-peptide pair data
    group_cols = ['domain', 'gene_name', 'pY_pos']
    data_cols = ['binding_call']
    temp_df_grouped = df[group_cols + data_cols].reset_index(drop=True).groupby(group_cols).agg(
        lambda x: True if any(x.astype(str).str.contains('binder')) else False)
    temp_df_grouped = temp_df_grouped.rename(columns={'binding_call': 'binder_in_rep_grp'}).reset_index()
    merge_df = df.copy().merge(temp_df_grouped, how='left', left_on=group_cols, right_on=group_cols)

    return merge_df


def old_evaluate_protein_functionality(df):
    # uses filename as proxy for 'run' in a single day -- Jones data follows this format
    # uses replicate groups with one or more binders as a determination that
    # the protein was, at some point, functional and can bind

    protein_funct_df = df.copy()[['filename', 'domain', 'domain_pos', 'binding_call', 'binder_in_rep_grp']].sort_values(
        ['domain', 'filename', 'domain_pos', 'binding_call', 'binder_in_rep_grp']).groupby(
        ['domain', 'filename', 'domain_pos']).agg(
        {'binding_call': lambda x: True if not any(x.astype(str).str.contains('binder')) else False,
         'binder_in_rep_grp': lambda x: True if any(x == True) else False})
    protein_funct_df = protein_funct_df.rename(
        columns={'binding_call': 'no_binder_in_run', 'binder_in_rep_grp': 'binder_in_other_run'})

    # protein is non-functional only if there is no binder in the run/day/filename but there are binders in other runs
    # in other words, mark whole days functional only if they contain verifiable false negatives.
    protein_funct_df[
        'protein_non_functional'] = protein_funct_df.no_binder_in_run & protein_funct_df.binder_in_other_run
    protein_funct_df = protein_funct_df.drop(['no_binder_in_run', 'binder_in_other_run'],
                                             axis=1)  # remove intermediate columns
    protein_funct_df = protein_funct_df.reset_index()
    merge_df = pd.merge(df, protein_funct_df, how='left', left_on=['domain', 'filename', 'domain_pos'],
                        right_on=['domain', 'filename', 'domain_pos'])
    merge_df['binding_call_functionality_evaluated'] = merge_df.binding_call
    merge_df.loc[merge_df.protein_non_functional == True, 'binding_call_functionality_evaluated'] = 'non-functional'

    return merge_df


def evaluate_protein_functionality(df):
    # uses filename as proxy for 'run' in a single day -- Jones data follows this format
    # uses replicate groups with one or more binders as a determination that
    # the protein was, at some point, functional and can bind at least some peptides

    # Annotates if there are binders in a given run (file), and if ever binds
    protein_funct_df = df.copy()[['filename', 'domain', 'domain_pos', 'binding_call', 'binder_in_rep_grp']].sort_values(
        ['domain', 'filename', 'domain_pos', 'binding_call', 'binder_in_rep_grp'])
    protein_funct_grouped_df = protein_funct_df.copy().groupby(
        ['domain', 'filename', 'domain_pos']).agg(
        {'binding_call': lambda x: True if any(x.astype(str).str.contains('binder')) else False}).rename(
        {'binding_call': 'binds_in_this_run'}, axis=1)
    protein_funct_df = protein_funct_grouped_df.copy().reset_index()
    protein_funct_grouped_df = protein_funct_grouped_df.reset_index()
    protein_funct_grouped_df = protein_funct_grouped_df[['domain', 'binds_in_this_run']]
    protein_funct_grouped_df = protein_funct_grouped_df.groupby('domain').agg({'binds_in_this_run': lambda x: x.max()})
    protein_funct_grouped_df = protein_funct_grouped_df.rename({'binds_in_this_run': 'ever_binds'},
                                                               axis=1).reset_index()
    protein_funct_df = pd.merge(protein_funct_df, protein_funct_grouped_df, how='left')

    # case 1: protein doesn't bind in this run, but bound in other runs -> this run nonfunctional
    # case 1: protein doesn't bind in this run, and never bound in any other runs
    #                                                       -> harder to call, this run is likely to be nonfunctional
    # case 3: protein does bind now -> functional
    # so the positive control for functionality is whether or not 1 or more binders are found in current run

    protein_funct_df[
        'protein_non_functional'] = ~protein_funct_df.binds_in_this_run

    protein_funct_df = protein_funct_df.drop(['binds_in_this_run', 'ever_binds'],
                                             axis=1)  # remove intermediate columns
    merge_df = pd.merge(df, protein_funct_df, how='left', left_on=['domain', 'filename', 'domain_pos'],
                        right_on=['domain', 'filename', 'domain_pos'])
    merge_df['binding_call_functionality_evaluated'] = merge_df.binding_call
    merge_df.loc[merge_df.protein_non_functional == True, 'binding_call_functionality_evaluated'] = 'non-functional'

    return merge_df


def analyze_replicates(df):
    start_time = time.time()

    # replicate analysis
    ms_utils.print_flush('\treplicate measurements  ' + str(df.shape[0]))

    # Make binding and Kd calls on replicate groups
    group_cols = ['domain', 'gene_name', 'pY_pos']
    df_grouped = df.reset_index().groupby(group_cols)
    sort_type = ['domain', 'gene_name', 'pY_pos']

    # get all existing binding calls: should be limited to: binder, nobinding, agregator, low-SNR
    bindingcall_list = sorted(df.binding_call.unique())
    functionalbinding_call_list = sorted(df.binding_call_functionality_evaluated.unique())
    rows_list = []

    for replicate_group_description, replicate_group_df in df_grouped:
        ###########################################################
        # Ronan-Naegle method to make calls for a replicate group
        ###########################################################
        # use binding_call_functionality_evaluated field for calls after protein functionality is evaluated
        # otherwise, use binding_call field for pre-functionality-modified assessment of binding

        row_dict = {}
        row_dict.update({'domain': replicate_group_description[0]})
        row_dict.update({'gene_name': replicate_group_description[1]})
        row_dict.update({'pY_pos': replicate_group_description[2]})

        # get counts for each of the different binding calls: binder, nobinding, agregator, low-SNR
        call_counts_dict = {bindingcall_text: (replicate_group_df.binding_call.values == bindingcall_text).sum() for
                            bindingcall_text in bindingcall_list}
        count_replicates = replicate_group_df.binding_call.shape[0]

        # binder call logic
        if call_counts_dict['binder'] > 0:
            row_dict.update({'binding_call': 'binder'})
        elif call_counts_dict['binder'] == 0 and call_counts_dict['nobinding'] > 0:
            row_dict.update({'binding_call': 'no-binding'})
        elif call_counts_dict['binder'] == 0 and call_counts_dict['nobinding'] == 0 and call_counts_dict[
            'aggregator'] > 0:
            row_dict.update({'binding_call': 'aggregator'})
        elif call_counts_dict['binder'] == 0 and call_counts_dict['nobinding'] == 0 and call_counts_dict[
            'aggregator'] == 0 and call_counts_dict['low-SNR'] > 0:
            row_dict.update({'binding_call': 'low-SNR'})
        else:
            row_dict.update({'binding_call': 'BAD LOGIC'})

        row_dict.update({'count_replicates': count_replicates})
        row_dict.update({'count_bind': call_counts_dict['binder']})
        row_dict.update({'count_nobind': call_counts_dict['nobinding']})
        row_dict.update({'count_agg': call_counts_dict['aggregator']})
        row_dict.update({'count_lowSNR': call_counts_dict['low-SNR']})

        # additional binding call revised to account for protein functionality
        # only affects non-binders, annotates false-negative non=binders to improve non-binding calls
        # uses the binding_call_functionality_evaluated field instead of the binding_call field

        # get counts for each of the different binding calls: binder, nobinding, agregator, low-SNR
        call_counts_dict = {
        bindingcall_text: (replicate_group_df.binding_call_functionality_evaluated.values == bindingcall_text).sum() for
        bindingcall_text in functionalbinding_call_list}
        count_replicates = replicate_group_df.binding_call_functionality_evaluated.shape[0]

        # binder call logic
        if call_counts_dict['binder'] > 0:
            row_dict.update({'functional_binding_call': 'binder'})
        elif call_counts_dict['binder'] == 0 and call_counts_dict['nobinding'] > 0:
            row_dict.update({'functional_binding_call': 'no-binding'})
        elif call_counts_dict['binder'] == 0 and call_counts_dict['nobinding'] == 0 and \
                call_counts_dict['aggregator'] > 0 and \
                call_counts_dict['low-SNR'] == 0 and \
                call_counts_dict['non-functional'] == 0:
            row_dict.update({'functional_binding_call': 'aggregator'})
        elif call_counts_dict['binder'] == 0 and call_counts_dict['nobinding'] == 0 and \
                call_counts_dict['aggregator'] == 0 and \
                call_counts_dict['low-SNR'] > 0 and \
                call_counts_dict['non-functional'] == 0:
            row_dict.update({'functional_binding_call': 'low-SNR'})
        elif call_counts_dict['binder'] == 0 and call_counts_dict['nobinding'] == 0 and \
                call_counts_dict['aggregator'] == 0 and \
                call_counts_dict['low-SNR'] == 0 and \
                call_counts_dict['non-functional'] > 0:
            row_dict.update({'functional_binding_call': 'nonfunctional'})
        else:
            row_dict.update({'functional_binding_call': 'mixed_agg_lowsnr_nonfunc'})
        row_dict.update({'count_nonfunctional': call_counts_dict['non-functional']})

        # make sure to keep track of variation in the measured peptides
        row_dict.update({'peptide_str': ','.join(sorted(replicate_group_df['peptide_str'].unique()))})
        row_dict.update(
            {'pep_seq_10mer_aligned': ','.join(sorted(replicate_group_df['pep_seq_10mer_aligned'].unique()))})
        row_dict.update({'domain_seq_align_gapless': ','.join(
            sorted(replicate_group_df['domain_seq_align_gapless'].fillna('').unique()))})
        row_dict.update(
            {'peptide_number': ','.join(sorted(replicate_group_df['peptide_number'].unique().astype('str')))})

        # create temp_df with only 'binders', then select the index of the row with the lowest Kd
        # use data from this row to populate row dictionary
        temp_df = replicate_group_df.loc[replicate_group_df.binding_call == 'binder']
        min_Kd_columns = [
            'filename',
            'plate_number',
            'domain_pos',
            'peptide_number',
            'peptide_description',
            'Fluorescence',
            'PeptideConc',
            'fit_Kd',
            'fit_Fmax',
            'fit_offset',
            'fit_residuals',
        ]

        if not temp_df.empty:
            index_value_of_minimum_kd_for_this_domain_peptide = \
            temp_df.loc[temp_df.fit_Kd == temp_df.fit_Kd.min()].index.values[0]
            for col in min_Kd_columns:
                row_dict.update({col: temp_df.loc[index_value_of_minimum_kd_for_this_domain_peptide, col]})
        else:
            for col in min_Kd_columns:
                row_dict.update({col: ''})

        ###########################################################
        # Simulate Jones method to make calls for a replicate group
        ###########################################################
        # create temp_df with only 'binders', filter for Kd < 20, then take the mean Kd
        temp_df = replicate_group_df.loc[replicate_group_df.simJones_binding_call == 'binder']
        temp_df = temp_df[temp_df.simJones_Kd <= 20]
        if not temp_df.empty:
            row_dict.update({'simJones_meanKd': temp_df.simJones_Kd.mean()})
            row_dict.update({'simJones_Kd_list': ','.join(temp_df.simJones_Kd.astype(str).values.tolist())})
        else:
            row_dict.update({'simJones_Kd': ''})
            row_dict.update({'simJones_Kd_list': ''})

        rows_list.append(row_dict)

    final_replicate_calls_df = pd.DataFrame(rows_list)

    # remove non-essential columns, and resort columns
    final_replicate_calls_df = final_replicate_calls_df[[
        'domain',
        'gene_name',
        'pY_pos',
        'count_replicates',
        'count_bind',
        'count_nobind',
        'count_agg',
        'count_lowSNR',
        'binding_call',
        'functional_binding_call',
        'count_nonfunctional',
        'fit_Kd',
        'simJones_meanKd',
        'simJones_Kd_list',
        'filename',
        'plate_number',
        'domain_pos',
        'peptide_number',
        'pep_seq_10mer_aligned',
        'domain_seq_align_gapless',
    ]]


    ms_utils.print_flush('\tReplicate analysis complete')
    print
    print '\t\tReplicate Group Summary'
    print '\t\t-----------------------'
    print '\t\tfinal replicate groups: ' + str(final_replicate_calls_df.shape[0])
    for value in final_replicate_calls_df.binding_call.unique():
        print '\t\t',value, sum(final_replicate_calls_df.binding_call == value)

    print '\t\t-----------------------'
    print '\t\tfunctional replicate groups: ' + str(final_replicate_calls_df.shape[0])
    for value in final_replicate_calls_df.functional_binding_call.unique():
        print '\t\t',value, sum(final_replicate_calls_df.functional_binding_call == value)

    return final_replicate_calls_df


def load_published_fits():
    """
    Loads the fits published by the original authors, removes PTB domains and dual-SH2 domains, converts naming
    conventions to those used in the manuscript, and returns the info as a pandas dataframe
     """
    fileName = 'origdata/JonesFPOriginalFits/Table S1 FP interaction data.csv'
    leung_df = pd.read_csv(fileName, sep=',', index_col=None)
    bad_receptor_name_list = ['AR']  # no raw AR data available
    leung_df = leung_df[~leung_df.Receptor.isin(bad_receptor_name_list)]
    duplicated_from_Hause_data_list = ['ErbB1', 'ErbB2', 'ErbB3', 'ErbB4']
    leung_df = leung_df[~leung_df.Receptor.isin(duplicated_from_Hause_data_list)]
    # make sure the Protein-Receptor-site combination is unique
    assert leung_df.groupby(['Protein', 'Receptor', 'pTyr site', 'Sequence']).count()['Mean KD (?M)'].all()

    fileName = 'origdata/JonesFPOriginalFits/journal.pone.0044471.s020.csv'
    hause_df = pd.read_csv(fileName, sep=',', index_col=None)
    hause_df = hause_df[~hause_df.Receptor.isnull()]  # removes empty lines
    hause_df['pTyr site'] = hause_df['pTyr site'].str.replace('-18mer','') # removes 18mer notations
    # make sure the Protein-Receptor-site combination is unique
    assert hause_df.groupby(['Protein', 'Receptor', 'pTyr site', 'Sequence']).count()['Mean KD (?M)'].all()

    header_rename_dict = {
        'Protein': 'domain',
        'Receptor': 'gene_name',
        'pTyr site': 'pY_pos',
        'Mean KD (?M)': 'pub_Kd_mean',
        'Queries': 'pub_num_replicates'
    }

    leung_df = leung_df.rename(columns=header_rename_dict)
    hause_df = hause_df.rename(columns=header_rename_dict) # this df has duplicates due to some 13-mer and 18-mer measurements both being reported

    # this section removes the 18-mers when there are also 13-mer measurements
    group_cols = ['domain', 'gene_name', 'pY_pos']
    temp_df = hause_df[group_cols + ['Length of Peptide']].groupby(group_cols).count()
    suspect_idx = temp_df[temp_df['Length of Peptide'] > 1].reset_index().drop('Length of Peptide', axis=1).apply(tuple,1)
    suspect_df = hause_df[hause_df[group_cols].apply(tuple, 1).isin(suspect_idx)].sort_values(group_cols)
    delete_idx = suspect_df[suspect_df['Length of Peptide'] == '18-mer'].index
    hause_df = hause_df.drop(delete_idx)

    cols = ['domain',
            'gene_name',
            'pY_pos',
            'pub_Kd_mean',
            'pub_num_replicates'
            ]
    leung_df = leung_df[cols]
    hause_df = hause_df[cols]


    df = pd.concat([hause_df, leung_df], axis=0)
#    assert df.groupby(['domain', 'gene_name', 'pY_pos']).count()['pub_Kd_mean'].all() # does this even work, no, b/c different Kds on Hause duplicates

    df.gene_name = df.gene_name.str.upper()
    df.gene_name = df.gene_name.replace({'ERBB1': 'EGFR'})

    df = df.sort_values(['domain', 'gene_name', 'pY_pos']).reset_index(drop=True)

    # remove PTB domains
    df = df[~df.domain.str.contains('-PTB')]
    df = df[~df.domain.isin(rdlf.get_PTB_domain_list())]

    # remove domains with double SH2 domains (have '-NC' in the domain name)
    df = df[~df.domain.str.contains('-NC')]

    # remove domains that don't match to current domain data (TNS-113 and TNS-114)
    df = df[~df.domain.isin(rdlf.get_bad_domain_list())]


    # PI3KR2-N has multiple entries, one with a blank trailing-space
    # some replicates have values for the non-space and the trailing-space, some just for the trailing-space
    # the published manuscript split these replicate groups and reported results for each of these 'names'
    # These should be corrected (for comparison purposes) by combining the replicates and weighting means
    # by the number of replicates
    # this must be fixed before the domain name change dictionary corrects the published data

    dup_reps_list = [
        ['GAB1',83],
        ['GAB1',183],
        ['GAB1',285],
        ['GAB1',373],
        ['GAB1',447],
        ['GAB1',472],
        ['GAB1',627],
        ['GAB1',689],
        ['KIT',721],
        ['MET',1313]
    ]

    for name, pos in dup_reps_list:
        mean01 = df[(df.domain == 'PIK3R2-N') & (df.gene_name == name) & (df.pY_pos == pos)].pub_Kd_mean.values[0]
        numreps01 = df[(df.domain == 'PIK3R2-N') & (df.gene_name == name) & (df.pY_pos == pos)].pub_num_replicates.values[0]
        mean02 = df[(df.domain == 'PIK3R2-N ') & (df.gene_name == name) & (df.pY_pos == pos)].pub_Kd_mean.values[0]
        numreps02 = df[(df.domain == 'PIK3R2-N ') & (df.gene_name == name) & (df.pY_pos == pos)].pub_num_replicates.values[0]
        revised_mean = (mean01/numreps01)+(mean02/numreps02)
        revised_numreps = numreps01+numreps02
        # edit the original entry with the correct name, weight means and sum replicates
        df.loc[(df.domain == 'PIK3R2-N') & (df.gene_name == name) & (df.pY_pos == pos), 'pub_Kd_mean'] = revised_mean
        df.loc[(df.domain == 'PIK3R2-N') & (df.gene_name == name) & (df.pY_pos == pos), 'pub_num_replicates'] = revised_numreps

        # drop entry with the trailing space
        df.drop(df[(df.domain == 'PIK3R2-N ') & (df.gene_name == name) & (df.pY_pos == pos)].index,inplace=True)

    solo_reps_list = [['KIT',672], ['KIT',730]] # only one entry for the _space version

    for name, pos in dup_reps_list:
        # no changes to means or replicates, just edit the name
        df.loc[(df.domain == 'PIK3R2-N ') & (df.gene_name == name) & (df.pY_pos == pos),'domain'] = 'PIK3R2-N'

    # fix typos and obsolete domain names to match our naming conventions
    domain_name_change_dict = rdlf.get_domain_name_change_dict()
    df = df.replace({'domain': domain_name_change_dict})
    df.pub_num_replicates = df.pub_num_replicates.astype(int)

    return df


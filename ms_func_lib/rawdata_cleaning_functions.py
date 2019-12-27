from __future__ import division
import pandas as pd
import numpy as np
from ms_func_lib import ms_utils



def expand_conc_and_data_into_individ_cols_JONES(df):
    ## expand concentrations and fp data into individual columns
    ms_utils.print_flush('\tExpanding concentrations and signal into individual columns')

    df = df.sort_values(['filename', 'plate_number', 'domain_pos', 'peptide_number', 'peptide_description', 'domain_name','bg','domain_conc'])
    # Data from the Leung paper has no values in the peptide_description field. Pandas goupby statements
    # now drop rows with Nans, so the field has to be moved to the agg section
    df = df.groupby(
        ['filename', 'plate_number', 'domain_pos', 'peptide_number', 'domain_name', 'bg']).agg(
        {'domain_conc': lambda x: tuple(x.unique()), 'FP': lambda x: tuple(x.unique()),
         'peptide_description': lambda x: x.unique()})
    temp_df = df.reset_index().copy()

    # split domain concentrations and FP measurements into their own columns
    fp_df = temp_df['FP'].apply(pd.Series)
    fp_df.columns = ['fp01', 'fp02', 'fp03', 'fp04', 'fp05', 'fp06', 'fp07', 'fp08', 'fp09', 'fp10', 'fp11', 'fp12']
    conc_df = temp_df['domain_conc'].apply(pd.Series)
    conc_df.columns = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10', 'c11', 'c12']
    df = pd.concat([temp_df, fp_df, conc_df], axis=1)

    return df


def correct_for_missing_values_JONES(df):
    """
    Some of the 12 concentrations had missing values. Curve fitting depends on all 12 values
    being present. Only the 11th and 12 column of fp measurements had null values.
    There are 11 fp11 null values and 908 fp12 null values out of 25129 records.
    The null values are corrected by copying the value from fp(-1) (the previous value)
    """

    ms_utils.print_flush('\tCorrecting missing values')
    # fixes 11 rows with missing 11th data points
    df.loc[df.fp11.isnull(), 'fp11'] = df.loc[df.fp11.isnull(), 'fp10']
    # fixes 908 replicates with only 11 measurements
    df.loc[df.fp12.isnull(), 'fp12'] = df.loc[df.fp12.isnull(), 'fp11']

    return df


def consolidate_signal_and_conc_cols(df):
    """
    After corrections to individual values, recompile the 12 concentration and FP signal values
    into their own respective columns.
    Remove the individual columns to reduce the size of the dataframe and be more readable.
    Convert the 'PeptideConc' and 'Fluorescence' fields to a numpy array of float64 values.
    Return the modified pandas dataframe.
    """
    ms_utils.print_flush('\tConsolidating concentrations and signals')

    signal_cols = ['fp01', 'fp02', 'fp03', 'fp04', 'fp05', 'fp06', 'fp07', 'fp08', 'fp09', 'fp10', 'fp11', 'fp12']
    conc_cols = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10', 'c11', 'c12']
    df['pepConcTxt'] = [','.join(map(str, x)) for x in df[conc_cols].astype('float64').values.tolist()]
    df['FluorTxt'] = [','.join(map(str, x)) for x in df[signal_cols].astype('float64').values.tolist()]
    df['PeptideConc'] = [np.asarray([float(y) for y in x]).astype('float64') for x in df['pepConcTxt'].str.split(',')]
    df['Fluorescence'] = [np.asarray([float(y) for y in x]).astype('float64') for x in df['FluorTxt'].str.split(',')]
    df.drop(signal_cols,axis=1, inplace=True)
    df.drop(conc_cols,axis=1, inplace=True)
    if 'FP' in df.columns:
        df.drop('FP', axis=1, inplace=True)
    if 'domain_conc' in df.columns:
        df.drop('domain_conc', axis=1, inplace=True)

    return df


def replace_signal_and_conc_cols(df):
    """
    Convert the 'PeptideConc' and 'Fluorescence' fields from str back to a numpy array of float64 values.
    Return the modified pandas dataframe.
    """
    ms_utils.print_flush('\tConsolidating concentrations and signals')

    df['PeptideConc'] = [np.asarray([float(y) for y in x]).astype('float64') for x in df['pepConcTxt'].str.split(',')]
    df['Fluorescence'] = [np.asarray([float(y) for y in x]).astype('float64') for x in df['FluorTxt'].str.split(',')]

    return df

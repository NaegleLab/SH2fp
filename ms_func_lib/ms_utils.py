from __future__ import division
import sys
import time
import pandas as pd

###############################################################
## Utilities
###############################################################

def print_flush(*args):

    for a in args:
        sys.stderr.write(str(a)+' ')
    sys.stderr.write('\n')
    sys.stderr.flush()


def add_peptide_and_domain_sequence(df):
    start_time = time.time()
    print_flush("\tAdding domain sequences.")

    # load peptide list and map peptide sequences to peptide number
    peptides_df = pd.read_csv('origdata/Jones FP master peptide list edited.csv', sep=',', low_memory=False,
                              index_col=None)
    peptides_df = peptides_df[['peptide_number', 'receptor', 'tyrosine number', 'edited_sequence']]
    peptides_df['receptor'] = peptides_df['receptor'].str.upper()

    # merge
    df.peptide_number = df.peptide_number.astype('int')
    peptides_df.peptide_number = peptides_df.peptide_number.astype('int')
    df = pd.merge(df, peptides_df, how='left', left_on=['peptide_number'], right_on=['peptide_number'])

    df['pep_seq_padded'] = df.edited_sequence.apply(
        lambda x: x.rjust(len(x) + 9 - x.find('y'), '-').ljust(18, '-'))
    df['pep_seq_10mer_aligned'] = df.pep_seq_padded.str.slice(5, 15)
    df = df.rename(columns={'edited_sequence': 'pep_seq_cleaned', 'domain_name': 'domain', 'receptor': 'gene_name', 'tyrosine number': 'pY_pos'})
    df['gene_name'] = df['gene_name'].str.upper()
    df['gene_name'] = df['gene_name'].str.replace('ERBB1','EGFR' )

    df = df.drop('pep_seq_padded', axis=1)
    df['peptide_str'] = df['gene_name'] + '_' + df['pY_pos'].astype(str)


    # add gapless-alignment domain sequences
    domain_seq_df = pd.read_csv('externaldata/SH2domain_gapless_alignment.csv', sep=',', index_col=None)
    df = pd.merge(df, domain_seq_df, how='left', left_on=['domain'], right_on=['domain'])
    df = df.rename(columns={'gapless_seq': 'domain_seq_align_gapless'})
    df['domain'] = df['domain'].str.upper()  # force uppercase on domain name


    print_flush("\t\tSequences added.", time.time() - start_time)
    return df


def revise_columns(df):
    df = df[[
        'domain',
        'gene_name',
        'pY_pos',
        'peptide_number',
        'peptide_description',
        'plate_number',
        'filename',
        'domain_pos',
        'bg',
        'pepConcTxt',
        'FluorTxt',
        'PeptideConc',
        'Fluorescence',
        'linear_slope',
        'linear_offset',
        'linear_rsq',
        'linear_AICc',
        'linear_resVar',
        'linear_success',
        'fo_Fmax',
        'fo_Kd',
        'fo_offset',
        'fo_rsq',
        'fo_AICc',
        'fo_resVar',
        'fo_success',
        'peptide_str',
        'pep_seq_cleaned',
        'pep_seq_10mer_aligned',
        'domain_seq_align_gapless'
    ]]

    return df


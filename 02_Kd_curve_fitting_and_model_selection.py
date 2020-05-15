#!/usr/bin/env python

import pandas as pd
import time
import os
from ms_func_lib import ms_utils
from ms_func_lib import rawdata_cleaning_functions as rdcf
from ms_func_lib import Kd_fitting_functions as kdff
"""
This script loads flat table Jones FP raw data from the previous step,
converts to a form where a line represents one replicate measurement for a particular domain-peptide pair
(i.e. all 12 concentrations for one domain-peptide pair from a single plate on each line).

Next, it prepares the data for curve fitting, then fits Kd curves, and outputs the results to a .csv file.
Curve fitting takes approximately 15 minutes, so if existing results of curve fitting are found, it will load
from the file rather than re-fit the data. Set the force_refit flag to true to override or remove the existing results.

Two fits are made: fitting a line with offset (linear), and fitting a first-order saturation curve (fo).

Linear fits add columns for slope, y-offset (offset), r-squared (rsq, as used by the original authors), Aikake Information
Criterion non-weighted (AIC) and weighted (AICc), the Durbin-Watson statistic (dw), residual variance (res_Var), and
a boolean flag indicating fitting success (success).
[linear_slope, linear_offset, linear_rsq, linear_AIC, linear_AICc, linear_dw, linear_resVar, linear_success]

First-order fits add a Fluroescence saturation value as predicted by the model (Fmax), the fitted
dissocation constant (Kd), and a y-offset value (offset). rsq, AIC, AICc, dw, resVar, and success are also added.
[fo_Fmax, fo_Kd, fo_offset, fo_rsq, fo_AIC, fo_AICc, fo_dw, fo_resVar, fo_success]

Peptide and domain sequences and annotations are added. The gene source for the peptide (receptor),
the tyrosine position, (tyrosine number), the original peptide sequence (pep_seq_orig), as well as formatted versions of
the peptide sequence (peptide_str, pep_seq_orig, pep_seq_cleaned, pep_seq_padded, pep_seq_10mer_aligned), are added. 
The uniprot ID (uniprot_id), SH2 domain containing protein (gene_name), and phosphotyrosine position (pY_pos) are noted. 
A gapless alignment for SH2 domains ('domain_seq_align_gapless', provided by Roman Sloutsky in 2016) is added for
further analysis.
"""


input_filename = '01_rawdata_loaded_from_expt_format.csv'
models_selected_filename = '02_fitted_Kd_and_model_selection_data.csv'
force_refit = False

total_start_time = time.time()

ms_utils.print_flush('Loading flat table FP rawdata and preparing for fitting.')
df = pd.read_csv(input_filename,sep=',',low_memory=False)
number_of_measurements = df.shape[0]
ms_utils.print_flush('\t\tIndividual measurements loaded: ' + str(number_of_measurements))

# expand concentrations and fp data into individual columns
df = rdcf.expand_conc_and_data_into_individ_cols_JONES(df)
# correct for missing values
df = rdcf.correct_for_missing_values_JONES(df)
# prepare signal and concentration columns for fitting
df = rdcf.consolidate_signal_and_conc_cols(df)
number_of_pairs = df.shape[0]
expected_number_of_pairs = number_of_measurements/12
ms_utils.print_flush('\t\tDomain-peptide pairs after reformatting: ' + str(number_of_pairs)+' Expected: ' + str(expected_number_of_pairs))
ms_utils.print_flush('Data ready for fitting. Total time elapsed: ',time.time()-total_start_time, 'seconds.')

df = ms_utils.add_peptide_and_domain_sequence(df)

# Fitting different models to the curves from the raw FP data.
fitting_df = kdff.fit_models_to_data(df.copy().reset_index())
fitting_df = ms_utils.revise_columns(fitting_df)
fitting_df = fitting_df.reset_index(drop=True)
ms_utils.print_flush('\t\tFits complete')


models_chosen_df = kdff.evaluate_fits(fitting_df)
models_chosen_df.to_csv(models_selected_filename,index=False)


print 'Replicate Summary'
print '-----------------'
for value in models_chosen_df.fit_method.unique():
    print value, sum(models_chosen_df.fit_method == value)
print
for value in models_chosen_df.binding_call.unique():
    print value, sum(models_chosen_df.binding_call == value)
print
for value in sorted(models_chosen_df[['fit_method', 'binding_call']].apply(tuple, axis=1).unique()):
    print value, models_chosen_df[(models_chosen_df.fit_method == value[0]) & (models_chosen_df.binding_call == value[1])].shape[0]
print

ms_utils.print_flush('Kd curves fitted to data. Model selected. Total time elapsed: ',time.time()-total_start_time, 'seconds.')
ms_utils.print_flush('Data exported to : ',models_selected_filename)
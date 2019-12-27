import pandas as pd
import time
from ms_func_lib import ms_utils

from ms_func_lib import rawdata_cleaning_functions as rdcf
from ms_func_lib import replicate_analysis_functions as raf


input_file = '02_fitted_Kd_and_model_selection_data.csv'
functionality_evaluated_filename = '03a_fitted_Kd_and_model_selection_data_with_functionality_evaluated.csv'
output_filename = '03b_grouped_data_including_published.csv'

total_start_time = time.time()

ms_utils.print_flush('\nAnalyzing replicate groups and identifying non-functional protein.')

df = pd.read_csv(input_file, low_memory=False)
# reformatting signal and concentration columns
df = rdcf.replace_signal_and_conc_cols(df)
# evaluate protein functionality, using groups with one or more binders as a proxy for minimally functional protein
df = raf.identify_replicate_groups_with_one_or_more_binders(df)
df = raf.evaluate_protein_functionality(df)
df.to_csv(functionality_evaluated_filename,index=False)
ms_utils.print_flush('Replicate groups analyzed and non-functional protein identified. Total time elapsed: ',time.time()-total_start_time, 'seconds.')
ms_utils.print_flush('Data exported to : ',functionality_evaluated_filename)

ms_utils.print_flush('\nBeginning replicate analysis')
# implement replicate analysis process as described in the manuscript
# also, simulate original publication fitting process
replicates_df = raf.analyze_replicates(df)
ms_utils.print_flush('Replicates analyzed, replicate calls made. Total time elapsed: ',time.time()-total_start_time, 'seconds.')


# load published results and add them to the replicates dataframe for comparisons
published_df = raf.load_published_fits()

# fixing dtypes to avoid errors in merging
replicates_df.domain = replicates_df.domain.astype(str)
published_df.domain = published_df.domain.astype(str)

replicates_df.gene_name = replicates_df.gene_name.astype(str)
published_df.gene_name = published_df.gene_name.astype(str)

replicates_df.pY_pos = replicates_df.pY_pos.astype(int)
published_df.pY_pos = published_df.pY_pos.astype(int)

# merged published results with our analysis
merge_df = pd.merge(replicates_df, published_df, how='left', left_on=['domain','gene_name','pY_pos'], right_on=['domain','gene_name','pY_pos'])
merge_df.to_csv(output_filename,index=False)

ms_utils.print_flush('Results merged with published data. Total time elapsed: ',time.time()-total_start_time, 'seconds.')
ms_utils.print_flush('Data exported to : ',output_filename)






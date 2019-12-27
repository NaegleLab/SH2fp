#!/usr/bin/env python 

import pandas as pd
import time
from ms_func_lib import ms_utils
from ms_func_lib import rawdata_load_functions as rdlf


"""
Script loads Jones FP raw data from the Hause-Jones and Leung-Jones papers.
Takes the raw plate format, and converts it to a flat table format.
Each line details a measurement from one well on a plate: i.e., a single FP value
for a domain peptide pair at a particular concentration.

rdlf.clean_domains function corrects/updates domain names to match Uniprot names,
removes data from unresolvable domains, and removes data from PTB domains.
Comment this function out to have domain names match the published data.
Note, that accuracy of subsequent analysis steps requires this function, as all domain
names need to be 'normalized' at this step. 
"""


rawdata_location = 'origdata/JonesFPrawdata/'
output_filename = '01_rawdata_loaded_from_expt_format.csv'



start_time = time.time()
## load data
ms_utils.print_flush('Loading FP raw data and formatting.')
df_list = []
df_list.append(rdlf.load_Hause_data(rawdata_location))
df_list.append(rdlf.load_Leung_data(rawdata_location))
df = pd.concat(df_list)
df = rdlf.clean_domains(df) # corrects/updates domain names, removes data from unresolvable domains and PTB domains.
df = df.sort_values(['filename', 'plate_number', 'domain_pos', 'peptide_number', 'peptide_description', 'domain_name','bg','domain_conc'])

df.to_csv(output_filename,index=False)

ms_utils.print_flush('Data loaded and reformatted. Total time elapsed: ',time.time()-start_time, 'seconds.')
ms_utils.print_flush('Data exported to : ',output_filename)


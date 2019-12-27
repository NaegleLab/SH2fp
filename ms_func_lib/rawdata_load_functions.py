from __future__ import division
import pandas as pd
import numpy as np
import time
from ms_func_lib import ms_utils
import itertools

def get_PTB_domain_list():
    PTB_list = ['APBB1-N','APBB1-C','APBB1-NC','APBB2-N','APBB2-C','APBB2-NC',
            'APBB3-N','APBB3-C','APBB3-NC','APBA1','APBA3','APBA2','DAB1','DAB2','MAPK8IP1',
            'MAPK8IP2','NUMBL','NUMB','RGS12','E129946','SHC1-PTB','E136111','DOK1','DOK2',
            'DOK4','DOK5','IRS1','IRS2','IRS4','FRS2','FRS3','RABGAP1','ANKS1','APPL','GULP1',
            'EPS8L2','TLN1','CCM2','RABGAP1L','EB-1','DOK5L']
    return PTB_list


def get_domain_name_change_dict():
    domain_name_change_dict = {
    'DAPP-1': 'DAPP1', # presumably a typo
    'INNPL1': 'INPPL1', # presumably a typo
    'INPPLN1': 'INPPL1', # presumably a typo
    'E185634': 'SHC4', # researched externally
    'E105251': 'SHD', # researched externally
    'E169291': 'SHE', # researched externally
    'E138606': 'SHF', # researched externally
    'E109111': 'SUPT6H', # researched externally
    'E1894100':'SH2D5',
    'E189410':'SH2D5',
    'TNS': 'TNS1', # presumably a typo
    'TENS1': 'TNS3', #TENS1 ID'd as Tensin-3 on external ErbB document, changed to TNS3 to match other data
    'CTEN': 'TNS4',
    'EAT2': 'SH2D1B',
    'PIK3R2-N ': 'PIK3R2-N', # extra space at end
    'TXN': 'TXK', # typo
    'SH2B':'SH2B1',
    'APS':'SH2B2',
    'LNK':'SH2B3',
    'SLNK':'SH2D6',
    'BRDG1':'STAP1',
    'BKS':'STAP2',
    'BRK':'PTK6',
    'SLP76':'LCP2',
    }
    return domain_name_change_dict


def get_bad_domain_list():
    bad_domain_list = ['TNS113','TNS114']
    return bad_domain_list


def get_bad_gene_name_list():
    bad_gene_name_list = ['CSF1R']
    return bad_gene_name_list


def get_duplicate_peptide_list():
    # DPQRyLVIQG, EGFR, 978
    # DPQRyLVIQG, ERBB4, 984
    # SyGVTVW, EGFR, 900
    # DVWSyGVTVW, ERBB2, 908
    # DVWSyGVTVW, ERBB3, 897
    # TIDVyMIMVK, EGFR, 944
    # TIDVyMIMVK, ERBB2, 952

    duplicate_peptide_list = [
        ('ERBB4', 984),
        ('ERBB2', 908),
        ('ERBB3', 897),
        ('ERBB2', 952)
    ]
    return duplicate_peptide_list


def clean_domains(df):
    """Fix domain related issues in the raw data.

    Takes a pandas dataframe, with a column 'domain_name',
    returns a pandas dataframe with text corrections in column domain_name.

    Fixes typos in domain names,
    Updates outdated domain names,
    Removes PTB domain data,
    Removes data from unresolvable domain names."""

    # remove data if domain is a PTB domain
    df = df[~df.domain_name.str.contains('-PTB')]
    df = df[~df.domain_name.isin(get_PTB_domain_list())]
    
    # remove domains with double SH2 domains (have '-NC' in the domain name)
    df = df[~df.domain_name.str.contains('-NC')]

    # correct problems with domain names and update old names
    domain_name_change_dict = get_domain_name_change_dict()
    df = df.replace({'domain_name': domain_name_change_dict})

    # remove domain data for which domain names cannot be resolved
    df = df[~df.domain_name.isin(get_bad_domain_list())]


    # uncomments to remove GST data
    #GST_domain_list = ['GST', 'GST-CRK', 'GST-CRKL', 'GST-Crk', 'GST-CrkL', 'GST-GRB2', 'GST-Grb2', 'GST-SRC', 'GST-Src']
    #df = df[~df.domain_name.isin(GST_domain_list)]

    return df


def load_Hause_data(rawdata_location):
    """Loads Hause data in plate format from raw files,
    and converts the data into a flate table format.
    Returns data in a Pandas dataframe.

    Function takes a directory name (located in the directory of the main script)
    as the raw data location, and appends it to the hard coded file names.

    """



    ms_utils.print_flush('\tLoading Hause FP raw data')
    local_start_time = time.time()
    working_data = pd.DataFrame()

    fileName_list = [
                    'FPInput_run1_4_10_09.csv',
                    'FPInput_run2_4_10_09.csv',
                    'FPInput_run3_4_21_09.csv',
                    'FPInput_run4_4_10_09.csv',
                    'FPInput_run5_4_10_09.csv'
    ]

    for fileName in fileName_list:
        filename_text = fileName.split('.')[0]
        PEPTIDE_COLUMN_1 = 24
        PEPTIDE_COLUMN_2 = 25
        INITIAL_CONCENTRATION_COLUMN = 26
        NUMBER_OF_MEASURED_CONCENTRATIONS = 12
        DATA_ROWS = 17
        PLATE_ROWS = 16
        PLATE_COLUMNS = 24

        tempinput = pd.read_csv(rawdata_location+'/'+fileName,sep=',',header=None, index_col=None,low_memory=False)
        number_of_plates = tempinput.iloc[0,0]
        concentrations = []

        concentraton_positions = list(INITIAL_CONCENTRATION_COLUMN+np.arange(NUMBER_OF_MEASURED_CONCENTRATIONS)*2)
        for i in concentraton_positions:
            concentrations.append(float(tempinput.iloc[1,i]))

        assert tempinput.iloc[2:,:].shape[0]/17 == number_of_plates
        tempinput = tempinput.iloc[2:,:]

        data_flattened_from_file = pd.DataFrame()
        interspersed_lines = pd.DataFrame()

        for plate_number in range(int(number_of_plates)):

            start_row = DATA_ROWS * plate_number
            end_row = (DATA_ROWS * (plate_number+1))

            data_start_column = 0
            data_end_column = PLATE_COLUMNS + data_start_column

            unique_domain_start_column = INITIAL_CONCENTRATION_COLUMN
            unique_domain_end_column = unique_domain_start_column +2 

            domain_start_column = unique_domain_start_column
            domain_end_column = PLATE_COLUMNS + domain_start_column

            unique_domain_names = tempinput.iloc[start_row:end_row-1,unique_domain_start_column:unique_domain_end_column].unstack().tolist()
            domain_names = tempinput.iloc[start_row:end_row-1,domain_start_column:domain_end_column].unstack().tolist()

            domain_name_position_numbers = np.arange(1,(PLATE_ROWS*2)+1)
            domain_name_position_numbers = np.tile(domain_name_position_numbers,12)

            peptides = tempinput.iloc[start_row:end_row-1,PEPTIDE_COLUMN_1:PEPTIDE_COLUMN_2+1]
            unique_peptide = tempinput.iloc[start_row,PEPTIDE_COLUMN_1:PEPTIDE_COLUMN_2+1]

            # reshape values into list format

            peptide_number = peptides[PEPTIDE_COLUMN_1]
            peptide_number = peptide_number.str.replace('Peptide','') # removes 'Peptide' from peptide number, if present
            peptide_number = peptide_number.str.replace('Pep','') # removes 'Pep' from peptide number, if present
            peptide_number = peptide_number.values.tolist()*PLATE_COLUMNS

            peptide_description = PLATE_COLUMNS*peptides[PEPTIDE_COLUMN_2].values.tolist()
            plate_values = tempinput.iloc[start_row:end_row-1,data_start_column:data_end_column].unstack()
            filename_info = PLATE_ROWS*PLATE_COLUMNS*[filename_text]
            concentration_info = [c for concentration_string in concentrations for c in 2*(PLATE_ROWS)*[concentration_string]]
            plate_number_info = list(itertools.repeat(plate_number+1,PLATE_ROWS*PLATE_COLUMNS))
            assert 384 == len(peptide_number) == len(peptide_description) == len(plate_values) == len(filename_info) == len(concentration_info) == len(domain_names) == len(plate_number_info) == len(domain_name_position_numbers)


            # handle the odd 17th row
            potential_background_values = tempinput.iloc[end_row-1,data_start_column:data_end_column].values
            interspersed_line_info = pd.DataFrame([[filename_text]+[plate_number+1]+[unique_peptide.iloc[0]]+[unique_peptide.iloc[1]]+potential_background_values.tolist()])
            interspersed_lines = interspersed_lines.append(interspersed_line_info)
            background = np.median(potential_background_values[0::2])
            background_info = 384 * [background]
            #corr_plate_values = plate_values - background
            flattened_data = zip(filename_info,plate_number_info,peptide_number,peptide_description,domain_names,domain_name_position_numbers,concentration_info,plate_values.tolist(),background_info)
            data_flattened_from_file = data_flattened_from_file.append(flattened_data)

        data_flattened_from_file.columns = ['filename','plate_number','peptide_number','peptide_description','domain_name','domain_pos','domain_conc','FP','bg']
        interspersed_lines.columns = ['filename','plate_number','peptide_number','peptide_description','col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17','col18','col19','col20','col21','col22','col23','col24']

        working_data = working_data.append(data_flattened_from_file)

    # fix the only plate with pre-subtracted background. Add average background of 140 to this plate.
    working_data.loc[working_data.filename == 'FPInput_run4_4_10_09','bg'] =140
    working_data.loc[working_data.filename == 'FPInput_run4_4_10_09','FP'] = working_data.loc[working_data.filename == 'FPInput_run4_4_10_09','FP'].apply(lambda x: x + 140)

    # correct typo in peptide number that is mismatched with peptide description all throughout data
    working_data.loc[(working_data.peptide_number == '133') & (working_data.peptide_description == 'ErbB3-pY1132-18mer'), 'peptide_number'] = '134'

    ms_utils.print_flush('\t\tHause FP raw data Loaded. Time elapsed: ',time.time()-local_start_time, 'seconds.')
    return working_data


def load_Leung_data(rawdata_location):
    ## Load Leung Raw FP data

    ms_utils.print_flush('\tLoading Leung FP raw data')
    local_start_time = time.time()
    working_data = pd.DataFrame()

    fileName_list = [
                    'Run1_KIT_badplatesremoved_KL_06_09_09.csv',
                    'Run1_MET_usethisfile_KL_06_09_09.csv',
                    'Run2_KIT_badplatesremoved_KL_06_09_09.csv',
                    'Run2_MET_badplatesremoved_KL_06_09_09.csv',
                    'Run3_KIT_usethisfile_KL_06_09_09.csv',
                    'Run3_MET_Gab_badplatesremoved_KL_06_09_09.csv',
                    'Run4Day3_Gab1_MET_usethisfile_KL_06_09_09.csv',
                    'Run4Day3_KIT_badplatesremoved_KL_06_09_09.csv',
                    'Run5_KIT_badplatesremoved_KL_06_09_09.csv',
                    'Run5_MET_Gab_badplatesremoved_KL_06_09_09.csv',
                    'Run6_KIT_badplatesremoved_KL_06_09_09.csv',
                    'Run6_MET_Gab_batplatesremoved_KL_06_09_09.csv'
                    ]

    for fileName in fileName_list:
        filename_text = fileName.split('.')[0]
        PEPTIDE_COLUMN_1 = 25 # there is no second peptide column in this Leung data
        PEPTIDE_COLUMN_2 = 25
        INITIAL_CONCENTRATION_COLUMN = 26
        NUMBER_OF_MEASURED_CONCENTRATIONS = 12
        DATA_ROWS = 17
        PLATE_ROWS = 16
        PLATE_COLUMNS = 24

        tempinput = pd.read_csv(rawdata_location+'/'+fileName,sep=',',header=None, index_col=None,low_memory=False)

        # fix peptide names with .txt extensions
        tempinput.ix[:,PEPTIDE_COLUMN_1] = tempinput.ix[:,PEPTIDE_COLUMN_1].str.replace('.txt','')

        number_of_plates = tempinput.iloc[0,0]
        concentrations = []

        concentraton_positions = list(INITIAL_CONCENTRATION_COLUMN+np.arange(NUMBER_OF_MEASURED_CONCENTRATIONS)*2)
        for i in concentraton_positions:
            concentrations.append(float(tempinput.iloc[1,i]))

        tempinput.dropna(how='all',inplace=True) # required to handle empty nan rows in data
        assert tempinput.iloc[2:,:].shape[0]/17 == number_of_plates
        tempinput = tempinput.iloc[2:,:]

        data_flattened_from_file = pd.DataFrame()
        interspersed_lines = pd.DataFrame()

        for plate_number in range(int(number_of_plates)):

            start_row = DATA_ROWS * plate_number
            end_row = (DATA_ROWS * (plate_number+1))

            data_start_column = 0
            data_end_column = PLATE_COLUMNS + data_start_column

            unique_domain_start_column = INITIAL_CONCENTRATION_COLUMN
            unique_domain_end_column = unique_domain_start_column +2 

            domain_name_position_numbers = np.arange(1,(PLATE_ROWS*2)+1)
            domain_name_position_numbers = np.tile(domain_name_position_numbers,12)

            domain_start_column = unique_domain_start_column
            domain_end_column = PLATE_COLUMNS + domain_start_column

            unique_domain_names = tempinput.iloc[start_row:end_row-1,unique_domain_start_column:unique_domain_end_column].unstack().tolist()
            domain_names = tempinput.iloc[start_row:end_row-1,domain_start_column:domain_end_column].unstack().tolist()

            peptides = tempinput.iloc[start_row:end_row-1,PEPTIDE_COLUMN_1:PEPTIDE_COLUMN_2+1] # only one peptide name here
            unique_peptide = tempinput.iloc[start_row,PEPTIDE_COLUMN_1:PEPTIDE_COLUMN_2+1] # only one peptide name here

            # reshape values into list format

            peptide_number = peptides[PEPTIDE_COLUMN_1]
            peptide_number = peptide_number.str.split('-').apply(lambda x: x[0]) # handles dashed peptide names found in Leung data only
            peptide_number = peptide_number.str.split('_').apply(lambda x: x[0]) # handles underscore peptide names found in Leung data only
            peptide_number = peptide_number.str.replace('Peptide','') # removes 'Peptide' from peptide number, if present
            peptide_number = peptide_number.str.replace('Pep','') # removes 'Pep' from peptide number, if present
            peptide_number = peptide_number.str.split('A').apply(lambda x: x[0]) # handles peptide names ending in 'A' found in Leung data only
            peptide_number = peptide_number.values.tolist()*PLATE_COLUMNS

            peptide_description = ['']*PLATE_ROWS*PLATE_COLUMNS # makes a vector of blank text, these values are missing in Leung
            plate_values = tempinput.iloc[start_row:end_row-1,data_start_column:data_end_column].unstack()
            filename_info = PLATE_ROWS*PLATE_COLUMNS*[filename_text]
            concentration_info = [c for concentration_string in concentrations for c in 2*(PLATE_ROWS)*[concentration_string]]
            plate_number_info = list(itertools.repeat(plate_number+1,PLATE_ROWS*PLATE_COLUMNS))
            assert 384 == len(peptide_number) == len(peptide_description) == len(plate_values) == len(filename_info) == len(concentration_info) == len(domain_names) == len(plate_number_info)

            # handle the odd 17th row
            potential_background_values = tempinput.iloc[end_row-1,data_start_column:data_end_column].values
            interspersed_line_info = pd.DataFrame([[filename_text]+[plate_number+1]+[unique_peptide.iloc[0]]+['']+potential_background_values.tolist()])
            interspersed_lines = interspersed_lines.append(interspersed_line_info)
            background = np.median(potential_background_values[0::2])
            background_info = 384 * [background]
            #corr_plate_values = plate_values - background
            flattened_data = zip(filename_info,plate_number_info,peptide_number,peptide_description,domain_names,domain_name_position_numbers,concentration_info,plate_values.tolist(),background_info)
            data_flattened_from_file = data_flattened_from_file.append(flattened_data)

        data_flattened_from_file.columns = ['filename','plate_number','peptide_number','peptide_description','domain_name','domain_pos','domain_conc','FP','bg']
        interspersed_lines.columns = ['filename','plate_number','peptide_number','peptide_description','col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17','col18','col19','col20','col21','col22','col23','col24']

        working_data = working_data.append(data_flattened_from_file)
    
    # correction for Run5_KIT_badplatesremoved_KL_06_09_09 - 26 - 663
    working_data.loc[working_data.FP==-1000,'FP']=140 # fix -1000 value

    ms_utils.print_flush('\t\tLeung FP raw data Loaded. Time elapsed: ',time.time()-local_start_time, 'seconds.')
    return working_data



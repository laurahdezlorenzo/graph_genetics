'''
Description
'''
import os
import numpy as np
import pandas as pd

import datetime as dt
from scipy import stats

def select_genetic_cohort(input_filename, output_filename):

    '''
    Description
    '''

    samples_filename = 'data/ADNI/field_names.txt'

    # Get samples names
    samples_file = open(samples_filename, 'r')
    samples_names = samples_file.read().split('\n')
    samples_file.close()
    del samples_names[0:79]
    samples_names = [i.upper() for i in samples_names]

    # Load original ADNIMERGE data
    adnimerge = pd.read_csv(input_filename, index_col='PTID', low_memory=False)

    # Select genetics cohort samples
    adnimerge_genetics = adnimerge.loc[samples_names]
    adnimerge_genetics.index = adnimerge_genetics.index.str.upper()

    # Preprocessing some data
    adnimerge_genetics['ABETA'].replace('>1700', 1700, inplace=True)
    adnimerge_genetics['PTAU'].replace('<8', 8, inplace=True)
    adnimerge_genetics['ABETA'].replace('<200', 200, inplace=True)
    adnimerge_genetics['ABETA'] = adnimerge_genetics['ABETA'].astype('float64')
    adnimerge_genetics['PTAU'] = adnimerge_genetics['PTAU'].astype('float64')

    # Replace NaN values
    adnimerge_genetics.replace('-4', np.nan, inplace=True)
    adnimerge_genetics.replace(r'(\d\:\d[0-9])|(\d[0-99]\:\d[0-99])|(\d[0-9]\:\d)|(\d\:\d)', np.nan, inplace=True, regex=True)

    print()
    print(adnimerge_genetics)
    adnimerge_genetics.to_csv(output_filename)

    return adnimerge_genetics

def biomarkers_processing(data):

    '''
    Process dataset.
    '''

    result = pd.DataFrame([])

    samples = list(set(data.index))
    # print(len(samples))

    columns = ['EXAMDATE', 'AGE', 'DX', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT',
                'APOE4', 'AV45', 'PIB', 'ABETA', 'TAU', 'PTAU', 'FDG']
    
    data = data[columns]

    for sample in samples:

        sample_rows = data.loc[sample]

        if type(sample_rows['EXAMDATE']) == str:
            most_recent_row = sample_rows.to_frame().T

            
        else:
            sample_rows = sample_rows.sort_values(by='EXAMDATE', ascending=True)

            sample_rows = sample_rows.replace({'AV45': {0: np.nan}}).ffill()
            sample_rows = sample_rows.replace({'PIB': {0: np.nan}}).ffill()
            sample_rows = sample_rows.replace({'ABETA': {0: np.nan}}).ffill()
            sample_rows = sample_rows.replace({'TAU': {0: np.nan}}).ffill()
            sample_rows = sample_rows.replace({'PTAU': {0: np.nan}}).ffill()
            sample_rows = sample_rows.replace({'FDG': {0: np.nan}}).ffill()

            sample_rows.loc[:,'EXAMDATE'] = pd.to_datetime(sample_rows.loc[:, 'EXAMDATE'], format='%Y-%m-%d')
            most_recent_exam = sample_rows['EXAMDATE'].max()
            most_recent_row = sample_rows[sample_rows['EXAMDATE'] == most_recent_exam]

        result = result.append(most_recent_row)
    
    result = result.drop(columns=['EXAMDATE'])
    
    return result

def create_positive_columns(data):

    data.loc[:, 'AV45+'] = np.where(
    data['AV45'] >= 1.11, 1, np.where(
    data['AV45'] <  1.11, 0, np.nan))

    data.loc[:, 'PIB+'] = np.where(
        data['PIB'] >= 1.27, 1, np.where(
        data['PIB'] <  1.27, 0, np.nan))

    data.loc[:, 'ABETA+'] = np.where(
        data['ABETA'] >= 192, 1, np.where(
        data['ABETA'] <  192, 0, np.nan))

    data.loc[:, 'TAU+'] = np.where(
        data['TAU'] >= 93, 1, np.where(
        data['TAU'] <  93, 0, np.nan))

    data.loc[:, 'PTAU+'] = np.where(
        data['PTAU'] >= 23, 1, np.where(
        data['PTAU'] <  23, 0, np.nan))

    data.loc[:, 'APOE4+'] = np.where(
        data['APOE4'] > 0, 1, np.where(
        data['APOE4'] == 0, 0, np.nan))

    return data

def timeseries_processing(data, biomarker):

    notna_biomarker = data[data[biomarker].notna()]
    samples = list(set(notna_biomarker.index))

    rows_biomarker = []
    counter = 0
    for s in samples:
        
        sample_data = notna_biomarker.loc[s, :]
        
        if sample_data.shape != (112,):
            
            counter += 1
            
            sample_data = sample_data.sort_values(by='Month', ascending=True)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(sample_data['Month'],
                                                                            sample_data[biomarker])
            
            rows_biomarker.append([s, slope, r_value])
            
    # print(f'Number of subjects with >= 2 {biomarker} measurements: {counter} out of {len(samples)}')

    biomarker_progression = pd.DataFrame(rows_biomarker,
                                            columns=['sample',
                                                    f'{biomarker}_slope',
                                                    f'{biomarker}_rvalue'])

    biomarker_progression.set_index(['sample'], inplace=True)
    
    return biomarker_progression

def biomarkers_score(data):

    av45_cohort = data[['AV45', 'APOE4', 'TAU', 'PTAU', 'ABETA']]
    av45_cohort = av45_cohort.dropna(thresh=5)
    # print(av45_cohort)

    pib_cohort = data[['PIB', 'APOE4', 'TAU', 'PTAU', 'ABETA']]
    pib_cohort = pib_cohort.dropna(thresh=5)
    # print(pib_cohort)

    # print('Number of patients with APOE4-AV45-ABETA-TAU-PTAU measurements:', av45_cohort.shape[0])
    # print('Number of patients with APOE4-PIB-ABETA-TAU-PTAU measurements:', pib_cohort.shape[0])

    av45_cohort['AV45_scoring'] = (av45_cohort['AV45']/av45_cohort['AV45'].mean()
                                    + av45_cohort['ABETA']/av45_cohort['ABETA'].mean()
                                    + av45_cohort['TAU']/av45_cohort['TAU'].mean()
                                    + av45_cohort['PTAU']/av45_cohort['PTAU'].mean())/4

    pib_cohort['PIB_scoring'] = (pib_cohort['PIB']/pib_cohort['PIB'].mean()
                                    + pib_cohort['ABETA']/pib_cohort['ABETA'].mean()
                                    + pib_cohort['TAU']/pib_cohort['ABETA'].mean()
                                    + pib_cohort['PTAU']/pib_cohort['ABETA'].mean())/4

    # print(av45_cohort)
    # print(pib_cohort)

    scoring = pd.concat([av45_cohort, pib_cohort], axis=1, sort=False)
    scoring.drop(columns=['AV45', 'PIB', 'APOE4', 'TAU', 'PTAU', 'ABETA'], inplace=True)
    
    return scoring


def main(input_filename, genetics_filename, output_filename):


    adnimerge_genetics = select_genetic_cohort(input_filename, genetics_filename)

    adnimerge_processed = biomarkers_processing(adnimerge_genetics)
    adnimerge_processed = create_positive_columns(adnimerge_processed)
    adnimerge_scoring = biomarkers_score(adnimerge_processed)

    tau_progression = timeseries_processing(adnimerge_genetics, 'TAU')
    ptau_progression = timeseries_processing(adnimerge_genetics, 'PTAU')
    fdg_progression = timeseries_processing(adnimerge_genetics, 'FDG')

    adnimerge_all = pd.concat([adnimerge_processed, adnimerge_scoring,
                                tau_progression, ptau_progression,
                                fdg_progression], axis=1, sort=False)
    print()
    print(adnimerge_all)
    adnimerge_all.to_csv(output_filename)


if __name__ == '__main__':

    os.chdir('ppa-graphs')
    main()
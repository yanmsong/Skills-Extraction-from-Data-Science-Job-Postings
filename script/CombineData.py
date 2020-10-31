import pandas as pd
from cleantext import clean
import re
import string
from DataCleaning import clean_name, clean_company, clean_location, clean_description
import random
random.seed = 6


def initial_process(data, source):
    # clean the datafram for name, company and location
    # this is for combining the dataset and remove duplicates
    data = data[['Name', 'Company', 'Salary', 'Location', 'Description']]
    data = data.drop_duplicates()
    data['Source'] = source
    data['Name'] = data['Name'].apply(clean_name)
    data['Company'] = data['Company'].apply(clean_company)
    data['Location'] = data['Location'].apply(clean_location)
    if source == 'Linkedin': # clean the description if the data is from linkedin
        data['Description'] = data['Description'].apply(clean_description)
    return data

def combine_data(data1, data2, remove_source):
    # find the jobs that are posted on both resources
    combine_data = pd.concat([data1, data2])
    dup_df = combine_data.groupby(['Company', 'Name', 'Location'])['Source'].nunique().loc[lambda x: x > 1].reset_index(name='count')
    dup_df = dup_df[['Company', 'Name', 'Location']]
    
    # obtain the index of jobs that are posted on both resources
    common = combine_data.reset_index().merge(dup_df,on=['Company', 'Name', 'Location'], how='right').set_index('index')
    common = common[common.Source == remove_source]
    common_index = common.index
    
    # remove the those duplicate jobs from one resource
    combine_data = combine_data[~combine_data.index.isin(common_index)]
    return combine_data.reset_index(drop=True)

def split_data(df, n_train=200, n_val=50):
    # split dataset to train, test, validation
    n_total = len(df)
    
    # generate random index for train test validation
    train_val_idx = random.sample(range(n_total), n_train+n_val)
    train_idx = train_val_idx[:n_train]
    val_idx = train_val_idx[n_train:]
    test_idx = list(set(range(n_total)) - set(train_idx) - set(val_idx))
    print("train: {} validation: {} test:{}".format(len(train_idx), len(val_idx), len(test_idx)))

    df['Train'] = 0
    df['Valid'] = 0
    df['Test'] = 0

    df['Train'].loc[train_idx] = 1
    df['Valid'].loc[val_idx] = 1
    df['Test'].loc[test_idx] = 1
    
    return df


if __name__ == '__main__':

    # load data
    link_ds = pd.read_csv('../data/linkedin_data_scientist.csv', index_col=0)
    link_da = pd.read_csv('../data/linkedin_data_analyst.csv', index_col=0)
    glass_ds = pd.read_csv('../data/glassdoor_data_scientist.csv', index_col=0)
    glass_da = pd.read_csv('../data/glassdoor_data_analyst.csv', index_col=0)
    indeed_ds = pd.read_csv('../data/indeed-ds-0407.csv')
    indeed_da = pd.read_csv('../data/indeed-da-0407.csv')

    # make sure the three dataset are in the same format
    indeed_ds = indeed_ds[['Job Title', 'Company', 'Salary', 'Location', 'Description']]
    indeed_ds = indeed_ds.rename(columns={'Job Title': 'Name'})

    indeed_da = indeed_da[['Job Title', 'Company', 'Salary', 'Location', 'Description']]
    indeed_da = indeed_da.rename(columns={'Job Title': 'Name'})

    # data scientist
    link_ds = initial_process(link_ds, 'Linkedin')
    glass_ds = initial_process(glass_ds, 'Glassdoor')
    combine_ds = combine_data(link_ds, glass_ds, 'Linkedin')

    indeed_ds = initial_process(indeed_ds, 'Indeed')
    combine_ds = combine_data(combine_ds, indeed_ds, 'Indeed')

    print(len(link_ds), len(glass_ds), len(indeed_ds), len(combine_ds))
    print(combine_ds.groupby('Source').count())
    
    combine_ds = split_data(combine_ds)
    combine_ds.to_csv('../data/master_ds.csv', index=False)


    # data analyst
    link_da = initial_process(link_da, 'Linkedin')
    glass_da = initial_process(glass_da, 'Glassdoor')
    combine_da = combine_data(link_da, glass_da, 'Linkedin')

    indeed_da = initial_process(indeed_da, 'Indeed')
    combine_da = combine_data(combine_da, indeed_da, 'Indeed')

    print(len(link_da), len(glass_da), len(indeed_da), len(combine_da))
    print(combine_da.groupby('Source').count())
    
    combine_da = split_data(combine_da)
    combine_da.to_csv('../data/master_da.csv', index=False)
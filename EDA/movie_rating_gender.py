import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import movie_data_formatter as mdf
from scipy.stats import ttest_ind, mannwhitneyu

def reformat_gender_identity(df):
    df['gender_identity'] = df['gender_identity'].apply(mdf.map_gender_identity)
    return df

def get_shrek_data(df):
    shrek_df = df[(df['movie_name'] == 'Shrek') & (df['year'] == '2001')]
    return shrek_df

def separate_gender_reviews(df):
    male_reviews = df[df['gender_identity'] == 'Male']['rating']
    female_reviews = df[df['gender_identity'] == 'Female']['rating']
    non_binary_reviews = df[df['gender_identity'] == 'Non_binary']['rating']
    return male_reviews, female_reviews, non_binary_reviews

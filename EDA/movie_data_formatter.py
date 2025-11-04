import pandas as pd
import argparse
import re
import numpy as np


"""
movie_data_formatter.py

Reformats the raw movie replication CSV data into a tidy, analysis-ready format.

Each row in the raw dataset represents one participant's responses across:
  - [1–400]: Movie ratings
  - [401–421]: Self-assessments on sensation-seeking behavior
  - [422–464]: Personality questions
  - [465–474]: Self-reported movie experience ratings
  - [475]: Gender identity
  - [476]: Only-child indicator
  - [477]: Movie enjoyed alone indicator

After reformatting, the output table has one row per (participant, movie) pair, 
with repeated participant-level attributes (personality, demographics, etc.).

You can also generate summary tables:
    - One row per participant with all their attributes
    - One row per movie review with participant ID, movie name, rating, and year
    - etc.
    
You can call this script from other EDA scripts to preprocess the data before analysis.
Makes it consistent and reusable across analyses.
"""

def clean_column_name(name):
    # remove anything within parentheses
    name = re.sub(r'\(.*?\)', '', name)
    
    # replace spaces with underscores and convert to lowercase
    name = name.strip().replace(' ', '_').lower()
    
    # remove characters that are not alphanumeric or underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    # remove multiple consecutive underscores
    name = re.sub(r'__+', '_', name)
    
    return name

def extract_movie_name(name):
    # object in parentheses is the year 
    year = re.search(r'\((\d{4})\)', name)
    if year:
        year = year.group(1)
    else:
        year = None

    # object before the year is the movie name
    name = name.split('(')[0].strip()
    return name, year

def map_gender_identity(value):
    gender_map = {
        1: 'Male',
        2: 'Female',
        3: 'Non-binary'
    }
    
    return gender_map.get(value, np.nan)

def define_base_columns(df):
    base = {
        "report_id": [],
        "movie_name": [],
        "rating": [], 
        "year": []
    }

    feature_names = [get_personality_features_names(df)] \
        + [get_movie_experience_names(df)] + [get_demographic_names(df)] + [get_sensation_seeking_feature_names(df)]

    for feature_set in feature_names:
        for feature in feature_set:
            base[feature] = []
            
    return base
    
def convert_to_table(df):
    base_processed_data = define_base_columns(df)
    
    # go line by line
    for _, row in df.iterrows():
        report_id = row['report_id']
        movie_ratings = get_movie_ratings(row)
        personality_features = get_personality_features(row)
        movie_experience = get_movie_experience(row)
        demographic_features = get_demographic_features(row)
        sensation_seeking_features = get_sensation_seeking_features(row)

        for i in range(len(movie_ratings['movie_name'])):
            base_processed_data['report_id'].append(report_id)
            base_processed_data['movie_name'].append(movie_ratings['movie_name'][i])
            base_processed_data['rating'].append(movie_ratings['rating'][i])
            base_processed_data['year'].append(movie_ratings['year'][i])
            
            for feature, value in personality_features.items():
                base_processed_data[feature].append(value)
                
            for feature, value in movie_experience.items():
                base_processed_data[feature].append(value)
                
            for feature, value in demographic_features.items():
                base_processed_data[feature].append(value)
                
            for feature, value in sensation_seeking_features.items():
                base_processed_data[feature].append(value)
                
    processed_df = pd.DataFrame(base_processed_data)
    
    # clean column names
    processed_df.columns = [clean_column_name(col) for col in processed_df.columns]
    return processed_df
    
def convert_to_reporter_table(df):
    # create a new dataframe with one row per reporter
    base_processed_data = define_base_columns(df)
    
    # exclude movie_name and rating since we want one row per reporter
    base_processed_data.pop('movie_name')
    base_processed_data.pop('rating')
    base_processed_data.pop('year')
    
    # go line by line
    for _, row in df.iterrows():
        report_id = row['report_id']
        personality_features = get_personality_features(row)
        movie_experience = get_movie_experience(row)
        demographic_features = get_demographic_features(row)
        sensation_seeking_features = get_sensation_seeking_features(row)

        base_processed_data['report_id'].append(report_id)
        
        for feature, value in personality_features.items():
            base_processed_data[feature].append(value)
            
        for feature, value in movie_experience.items():
            base_processed_data[feature].append(value)
            
        for feature, value in demographic_features.items():
            base_processed_data[feature].append(value)
            
        for feature, value in sensation_seeking_features.items():
            base_processed_data[feature].append(value)

    processed_df = pd.DataFrame(base_processed_data)
    return processed_df

def convert_to_movie_reviews_table(df):
    base_processed_data = {
        "report_id": [],
        "movie_name": [],
        "rating": [],
        "year": []
    }
    
    # go line by line
    for _, row in df.iterrows():
        report_id = row['report_id']
        movie_ratings = get_movie_ratings(row)

        base_processed_data['report_id'].extend([report_id] * len(movie_ratings['movie_name']))
        base_processed_data['movie_name'].extend(movie_ratings['movie_name'])
        base_processed_data['rating'].extend(movie_ratings['rating'])
        base_processed_data['year'].extend(movie_ratings['year'])

    processed_df = pd.DataFrame(base_processed_data)
    return processed_df
    
    
#### GETTERS FOR DIFFERENT FEATURES ####  
# ---- Clean Movie Names ---- #
def add_report_id(df):
    # insert report_id as the last column
    df.insert(len(df.columns), 'report_id', range(1, len(df) + 1))
    df['report_id'] = df['report_id'].astype(int)
    return df

# ---- Movie Ratings ---- #
def get_movie_ratings(row):
    movie_names = get_movie_names(row)
    
    # clean movie names
    year, name = [], []
    for movie in movie_names:
        clean_name, clean_year = extract_movie_name(movie)
        name.append(clean_name)
        year.append(clean_year)
    
    movie_ratings = {
        'movie_name': name,
        'year': year,
        'rating': [row[movie] for movie in movie_names]
    }
    return movie_ratings

def get_movie_names(df):
    if isinstance(df, pd.Series):
        return df.index[0:400].tolist()
    return df.columns[0:400].tolist()

# ---- Sensation Seeking Features ---- #
def get_sensation_seeking_features(row):
    sensation_seeking_features = get_sensation_seeking_feature_names(row)
    sensation_seeking_features = {
        feature: row[feature] for feature in sensation_seeking_features
    }
    return sensation_seeking_features

def get_sensation_seeking_feature_names(df):
    if isinstance(df, pd.Series):
        return df.index[400:421].tolist()
    return df.columns[400:421].tolist()

# ---- Personality Features ---- #
    
def get_personality_features(row):
    personality_features = get_personality_features_names(row)
    personality_features = {
        feature: row[feature] for feature in personality_features
    }
    return personality_features

def get_personality_features_names(df):
    if isinstance(df, pd.Series):
        return df.index[421:464].tolist()
    return df.columns[421:464].tolist()

# ---- Movie Experience  ---- #

def get_movie_experience(row):
    movie_experience = {
        feature: row[feature] for feature in get_movie_experience_names(row)
    }
    return movie_experience

def get_movie_experience_names(df):
    if isinstance(df, pd.Series):
        return df.index[464:474].tolist()
    return df.columns[464:474].tolist()


# ---- Demographic Features ---- #
def get_demographic_names(df):
    if isinstance(df, pd.Series):
        return df.index[474:477].tolist()
    return df.columns[474:477].tolist()

def get_demographic_features(row):
    demographic_features = {
        feature: row[feature] for feature in get_demographic_names(row)
    }
    return demographic_features

def get_gender_identity(row):
    return row[474]

def get_only_child(row):
    return row[475]

def get_enjoyed_alone(row):
    return row[476]

def clean_data(df):
    df = add_report_id(df)
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("output_path", type=str, help="Path to the output CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)
    df = clean_data(df)
    processed_data = convert_to_table(df)
    processed_data.to_csv(args.output_path + "rating_by_movie.csv", index=False)
    
    reporter_data = convert_to_reporter_table(df)
    reporter_data.to_csv(args.output_path + "reporter_data.csv", index=False)
    
    movie_reviews_data = convert_to_movie_reviews_table(df)
    movie_reviews_data.to_csv(args.output_path + "movie_reviews.csv", index=False)

if __name__ == "__main__":
    main()

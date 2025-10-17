import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import movie_data_formatter as mdf

def get_movie_aggregation(df):
    movie_group = df.groupby('movie_name')
    movie_df = movie_group.agg(
        count_num_reviews = pd.NamedAgg(column='rating', aggfunc='count'),
        average_rating = pd.NamedAgg(column='rating', aggfunc='mean'),
        year = pd.NamedAgg(column='year', aggfunc='first')
    ).reset_index()
    
    # median split for popularity
    movie_df['popularity_level'] = median_split(movie_df['count_num_reviews'])
    
    # meadian split for year 
    movie_df['year'] = pd.to_numeric(movie_df['year'], errors='coerce')
    movie_df['year_level'] = median_split(movie_df['year'], high_label='new', low_label='old')
    
    return movie_df

def median_split(series, high_label='high', low_label='low'):
    median_value = series.median()
    return series.apply(lambda x: high_label if x >= median_value else low_label)

def plot_movie_popularity(movie_df, show=True):
    # x = count_num_reviews
    # y = average_rating
    top_movies = movie_df.sort_values(by='count_num_reviews', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.scatter(top_movies['count_num_reviews'], top_movies['average_rating'], alpha=0.6)
    plt.title('Movie Popularity: Number of Reviews vs Average Rating')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Average Rating')
    plt.grid(True)
    if show:
        plt.show()
        
    return plt

def plot_popular_median_split(popular_df, show=True):
    plt.figure(figsize=(10, 6))
    colors = {'high': 'blue', 'low': 'orange'}
    for level in popular_df['popularity_level'].unique():
        subset = popular_df[popular_df['popularity_level'] == level]
        plt.scatter(subset['count_num_reviews'], subset['average_rating'], 
                    c=colors[level], label=level, alpha=0.6)
    plt.title('Movie Popularity with Median Split')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Average Rating')
    plt.legend(title='Popularity Level')
    plt.grid(True)
    if show:
        plt.show()
        
    return plt

def plot_year_median_split(movie_df, show=True):
    plt.figure(figsize=(10, 6))
    colors = {'new': 'green', 'old': 'red'}
    for level in movie_df['year_level'].unique():
        subset = movie_df[movie_df['year_level'] == level]
        plt.scatter(subset['count_num_reviews'], subset['average_rating'], 
                    c=colors[level], label=level, alpha=0.6)
    plt.title('Movie Popularity by Year Median Split')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Average Rating')
    plt.legend(title='Year Level')
    plt.grid(True)
    if show:
        plt.show()
        
    return plt

def plot_boxplots_year_level(movie_df, show=True):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='year_level', y='average_rating', data=movie_df)
    plt.title('Boxplots of Average Rating by Year Level')
    plt.xlabel('Year Level')
    plt.ylabel('Average Rating')
    plt.grid(True)
    if show:
        plt.show()
    return plt

def plot_boxplots_popularity_level(movie_df, show=True):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='popularity_level', y='average_rating', data=movie_df)
    plt.title('Boxplots of Average Rating by Popularity Level')
    plt.xlabel('Popularity Level')
    plt.ylabel('Average Rating')
    plt.grid(True)
    if show:
        plt.show()
    return plt

def main():
    args = mdf.parse_args()
    df = pd.read_csv(args.input_file)
    df = mdf.clean_data(df)
    processed_data = mdf.convert_to_movie_reviews_table(df)
    popular_data = get_movie_aggregation(processed_data)
    
    plot_movie_popularity(popular_data, show=False)
    plt.savefig(args.output_path + "movie_popularity.png")
    
    plot_popular_median_split(popular_data, show=False)
    plt.savefig(args.output_path + "movie_popularity_median_split.png")

    plot_year_median_split(popular_data, show=False)
    plt.savefig(args.output_path + "movie_year_median_split.png")
    
    plot_boxplots_year_level(popular_data, show=False)
    plt.savefig(args.output_path + "boxplot_year_level.png")

    plot_boxplots_popularity_level(popular_data, show=False)
    plt.savefig(args.output_path + "boxplot_popularity_level.png")


if __name__ == "__main__":
    main()
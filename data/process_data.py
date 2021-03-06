import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load datasets and merge them into one 

    INPUT 
    @messages_filepath   : the file path of `messages` file
    @categories_filepath : the file path of `categories` file

    OUTPUT
    @df : a pandas dataframe which merges the data of the file provided
    """

    # load datasets
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories,on="id")
    return df


def clean_data(df):
    """
    Clean the original data for further modeling

    INPUT
    @df : the pandas dataframe that produced from `load_data` function

    OUTPUT
    @df : a pandas dataframe that has been cleaned 
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories
    extract_colname   = lambda x: x[:-2]
    category_colnames = row.map(extract_colname)
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    extract_number_to_int = lambda x: int(x[-1])
    categories = categories.applymap(extract_number_to_int)
    # replace the 2 entries by 1
    categories['related'][categories['related']==2]=1
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # remove duplicates
    df = df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    """
    Save the cleaned data into database
    
    INPUT
    @df : a pandas dataframe of cleaned data (produced from the `clean_data` function)
    @database_filename : the filename (possibly with file path) of database
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("DisasterResponse", engine, index=False,if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
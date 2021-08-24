""" 1.Import libraries and load datasets """
import pandas as pd
from sqlalchemy import create_engine

messages   = pd.read_csv("disaster_messages.csv")
categories = pd.read_csv("disaster_categories.csv")


""" 2. Split categories into separate category columns """
# merge datasets
df = pd.merge(messages,categories,on="id")

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(";",expand=True)
# select the first row of the categories dataframe
row = categories.iloc[0]
# extract a list of new column names for categories
extract_colname   = lambda x: x[:-2]
category_colnames = row.map(extract_colname)
# rename the columns of `categories`
categories.columns = category_colnames


""" 3. Convert category values to just numbers 0 or 1 """
extract_number_to_int = lambda x: int(x[-1])
categories = categories.applymap(extract_number_to_int)


""" 4. Replace categories column in df with new category columns"""
# drop the original categories column from `df`
df = df.drop('categories', axis=1)
# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df,categories],axis=1)
# remove duplicated
df = df.drop_duplicates()


""" 5. Save the clean dataset into an sqlite database """
engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('DisasterResponse', engine, index=False)
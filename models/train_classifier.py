#%%
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.utils.fixes import loguniform

def load_data(database_path):
    """ 
    load data from database 
    """
    engine = create_engine('sqlite:///' + database_path)
    df = pd.read_sql_table('DisasterResponse',engine)

    X = df['message']
    Y = df.drop(['message','id','original','genre'], axis=1)
    # fill NaN values & drop columns with only one value
    Y = Y.fillna(0).drop('child_alone',axis=1)
    return X,Y



def tokenize(text):
    """ 
    Tokenization function for the text data 
    """
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    new_text = " ".join(tokens)
    return new_text



def build_model_cv(classifier,params,search_method="grid"):
    """
    Build a ML pipeline using TF-IDF and a classifier chosen by user

    INPUT:
    classifier    - an instance of a ML classifier 
    params        - range of parameters in form of dictionary, for GridSearchCV
    search_method - {"grid","random"}. 
                    "grid" refers to GridSearchCV; "random" refers to "RandomizedSearchCV"

    OUTPUT:
    a grid search CV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(classifier))
    ])

    if search_method == "grid":
        cv = GridSearchCV(pipeline,param_grid=params)
    elif search_method == "random":
        cv = RandomizedSearchCV(pipeline,param_distributions=params)

    return cv




def main():
    database_path = "./data/DisasterResponse.db"
    classifier = SVC()
    params = {
        {
            'kernel': ['poly'],
            'degree': np.arange(1,9),
            'gamma' : loguniform(1e-4, 1e-1),
            'C'     : loguniform(1e-3, 1e1)
        },

        {
            'kernel': ['rbf'],
            'gamma' : loguniform(1e-4, 1e-1),
            'C'     : loguniform(1e-4, 1e1)
        }
    }
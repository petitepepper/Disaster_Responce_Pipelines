import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import joblib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report,accuracy_score


def load_data(database_filepath):
    """ 
    load data from database 
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)

    X = df['message']
    Y = df.drop(['message','id','original','genre'], axis=1)
    
    # drop columns that has only one value
    Y = Y.drop('child_alone',axis=1)
    # drop NaN
    Y.dropna(inplace=True)

    # the name will be used in further steps
    category_names = Y.columns.values

    return X,Y,category_names


def tokenize(text):
    """ 
    Tokenization function for the text data 
    """
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    #Regex to find urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Finds all urls from the provided text
    detected_urls = re.findall(url_regex, text)
    #Replaces all urls found with the "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    #new_text = " ".join(tokens)
    return clean_tokens


def build_model(search_method="grid"):
    """
    Build a ML pipeline using TF-IDF and a classifier chosen by user

    INPUT:
    search_method - {"grid","random"}. 
                    "grid" refers to GridSearchCV; "random" refers to "RandomizedSearchCV"

    OUTPUT:
    a grid search CV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(SVC()))
    ])

    params = {
        'clf__estimator__kernel': ['rbf','linear'],
        'clf__estimator__C'     : [1e-3,1e-2,1e-1,1,10]
    }


    if search_method == "grid":
        cv = GridSearchCV(
            pipeline,
            param_grid=params,
            cv=3
        )
    elif search_method == "random":
        cv = RandomizedSearchCV(
            pipeline,
            param_distributions=params,
            cv=3
        )

    return cv


def evaluate_model(model, X_test, Y_test, category_names,print_result=True):
    y_pred = model.predict(X_test)
    n_class = len(category_names)
    avg = 0
    for i in range(n_class):
        accuracy = accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])
        avg += accuracy
        if print_result:
            print("Category:",category_names[i])
            print("Accuracy: %.4f"%(accuracy))
            print(classification_report(Y_test.iloc[:, i].values, y_pred[:,i]))
            print("\n")

    avg /= n_class
    print(f"Average of accuracy:{avg}")
    return avg


def save_model(model, model_filepath):
    joblib.dump(model,model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
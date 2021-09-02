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
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report,accuracy_score
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

    # the name will be used in further steps
    class_names = Y.columns.values

    onehot_encoder = OneHotEncoder(sparse = False)
    Y = onehot_encoder.fit_transform(Y)

    return X,Y,class_names



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
        cv = GridSearchCV(
            pipeline,
            param_grid=params,
            n_jobs=-1,cv=3
        )
    elif search_method == "random":
        cv = RandomizedSearchCV(
            pipeline,
            param_distributions=params,
            n_jobs=-1,cv=3
        )

    return cv


def model_evaluation(model,X_test,y_test,class_names,print_result=True):
    y_pred = model.predict(X_test)
    n_class = len(class_names)
    avg = 0
    for i in range(n_class):
        accuracy = accuracy_score(y_test[:,i], y_pred[:,i])
        avg += accuracy
        if print_result:
            print("Category:",class_names[i])
            print("Accuracy: %.4f"%(accuracy))
            print(classification_report(y_test[:,i], y_pred[:,i]))
            print("\n")

    avg /= n_class
    print(f"Average of accuracy:{avg}")
    return avg



#%%
def main():
    # load data
    database_path = "../data/DisasterResponse.db"
    X,Y,class_names = load_data(database_path)
    X_train,X_test,y_train,y_test = train_test_split(X,Y)

    # set classifier and parameters
    
    # Random Forest
    classifier = RandomForestClassifier()
    params = {
        'clf__estimator__n_estimators' : [50,100,200],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__criterion': ['entropy', 'gini']
    }
    
    """
    # SVM
    classifier = SVC()
    params = {
        'clf__estimator__kernel': ['rbf'],
        'clf__estimator__gamma' : loguniform(1e-4, 1e-1),
        'clf__estimator__C'     : loguniform(1e-4, 1e1)
    }
    """

    print("Building model...")
    model = build_model_cv(classifier,params)

    print("Training model...")
    model.fit(X_train, y_train)

    model_evaluation(model,X_test,y_test,class_names)

    # save model
    print("Saving model...")
    joblib.dump(model,'classifier.pkl')
    print("Model saved successfully!")

    
if __name__ == '__main__':
    main()

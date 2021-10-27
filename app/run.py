import json
import plotly
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier_svc.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories = df.drop(["id","message","original","genre"],axis=1)
    
    category_counts_df = categories.sum(axis=0)
    category_counts = list(category_counts_df.values)
    category_names  = list(category_counts_df.index.values)
    
    requests_counts  = categories[categories['request']==1].drop(["request","offer"],axis=1).sum()
    offers_counts    = categories[categories['offer']==1].drop(["request","offer"],axis=1).sum()
    hist_data    = [requests_counts.values,offers_counts.values]
    group_labels = ["Request","Offer"]

    # create visuals
    graphs = [
        # Figure 1 : Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },

        # Figure 2 : Category Counts
        {
            'data':[
                px.treemap(
                    names  = category_names,
                    values = category_counts
                )
            ],
            'layout':{
                'title' : 'Count of Message Categories',
            }
        },

        # Figure 3 : Requests & Offers Distribution 
        {
            'data':[ff.create_distplot(hist_data,group_labels)],
            'layout':{
                'title':'Distribution of Requests and Offers',
                'xaxis': {'title':'category'},
                'yaxis': {'title':'counts'}
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
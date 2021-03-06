# Disaster_Response_Pipelines
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with [Figure Eight](https://appen.com/). The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.





## Package

In python `3.x`, the following packages are used in this project:

- numpy
- pandas
- sqlalchemy
- nltk
- sklearn

Notice that, when using `nltk`, you may need to download some packages using the following code:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```





## Document hierarchy

In this repository, document hierarchy are shown as follow:

```
├── app
│   ├── run.py--------------------------# FLASK FILE THAT RUNS APP
│   └── templates
│       ├── go.html---------------------# CLASSIFICATION RESULT PAGE OF WEB APP
│       └── master.html-----------------# MAIN PAGE OF WEB APP
│
├── data
│   ├── DisasterResponse.db-------------# DATABASE TO SAVE CLEANED DATA TO
│   ├── disaster_categories.csv---------# DATA TO PROCESS
│   ├── disaster_messages.csv-----------# DATA TO PROCESS
│   ├── ETL Pipeline Preparation.ipynb--# DATA ETL DEMOSTRATION (same as `process_data.py`)
│   └── process_data.py-----------------# PERFORMS ETL PROCESS
│
├── models
│   ├── ML Pipeline Preparation.ipynb---# ML PIPELINE DEMOSTRATION (same as `train_classifier.py`)
│   └── train_classifier.py-------------# PERFORMS CLASSIFICATION TASK
│
├── src---------------------------------# IMAGES USED IN　README　FILE

```

 



## Data Understanding

The dataset provided by [Figure Eight](https://www.figure-eight.com/) contains over 30000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, superstorm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data is separated into two files *disaster_messages.csv* and *disaster_categories.csv* , which can be found in the folder `data/`.

- In the *disaster_messages.csv*  file, original messages in different languages have been translated to English and stored in the column `message`

  <img src=".\src\data_messages_origin.png" style="zoom:58%;" />

- These messages are divided into 36 categories related to disaster response, and saved in *disaster_categories.csv*

<img src=".\src\data_category_origin.png" style="zoom:45%;" />





## ETL Pipeline

After viewing the original data, it can be easily found that the  `categories` column in *disaster_categories.csv* file should be cleaned. To be specific, it should  be separated into different columns, of which each represents a category of message. Thus the following steps will be taken:

1. Extract category names and set them as column names

   To extract the names, we can leverage the `split` function (take `;` as separator), and then remove the last two characters by using the slice.

   <img src=".\src\data_step1.png" style="zoom:50%;" />

2. Keep the last character of each element 

   <img src=".\src\data_step2.png" style="zoom:50%;" />

3. Merge two sets of data, which are the disaster messages and the disaster categories

4. Remove duplicated records

5. Save data in database with the `sqlalchemy` library



**Execute the python file `process_data.py` in your terminal as following (in the '/data' path) :**

`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

<img src=".\src\process_data_command_line.png" style="zoom:50%;" />





## Data Modeling

After the ETL operations, the new data we have obtained is as follow:

<img src=".\src\data_new.png" style="zoom:40%;" />



### Preprocessing

Before modeling, we still need to check if the data meet the following requirements:

- does not have `NaN` value
- for each label (category), there will not be only one value in the samples

Next, we will move on to the most interesting part!



### NLP

In order to achieve better performance of the classifier later, we need to **tokenize** each text. The following operations can be performed:

- remove punctuation
- convert all to lowercase
- remove stop words (i.e. words that usually have no meaning, e.g. "a", "the", "is")
- lemmatizer words (convert the word to its original form)

Then, a **TF-IDF**(Term Frequency - Inverse Document Frequency) will be implemented. It is a word frequency based method which helps to identify the words that distinguish each text from others.



### Modeling

Unlike the case we see most often, here, a message can have **multiple labels** (e.g. the message with id=7). In *sklearn*, the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) will be quite helpful for predicting multiple target variables. 

> Note that it is necessary to check that the format of the data used as input and output meets the requirements



To obtain satisfactory results, we can use <u>random search</u> and/or <u>grid search</u> to find the best combination of hyperparameters.



**Execute the python file `train_classifier.py` in your terminal as following (in the '/models' path) :**

`python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

<img src=".\src\training_model.png" style="zoom:50%;" />



## Evaluation

For testing multi-label classification models, the accuracy of the model can be calculated using <u>Hamming loss</u>. Of course, we can also use the built-in  `classification_report` method in *sklearn*,  however, it should be used once for each tag.

Here, we tried SVM and Naive Bayes. The best results come from the random forest model, which achieves an average accuracy of 96%, while the other is only about 82%. 

You can check the evaluation report in the file "/models/ML Pipeline Preparation.ipynb"

> ```
> Category: related
> Accuracy: 0.8995
>               precision    recall  f1-score   support
> 
>            0       0.85      0.68      0.75      1496
>            1       0.91      0.97      0.94      5058
> 
>     accuracy                           0.90      6554
>    macro avg       0.88      0.82      0.85      6554
> weighted avg       0.90      0.90      0.90      6554
> 
> 
> 
> Category: request
> Accuracy: 0.9246
>               precision    recall  f1-score   support
> 
>            0       0.94      0.98      0.96      5448
>            1       0.85      0.67      0.75      1106
> 
>     accuracy                           0.92      6554
>    macro avg       0.89      0.82      0.85      6554
> weighted avg       0.92      0.92      0.92      6554
> 
> 
> 
> Category: offer
> Accuracy: 0.9954
>               precision    recall  f1-score   support
> 
>            0       1.00      1.00      1.00      6524
>            1       0.00      0.00      0.00        30
> 
>     accuracy                           1.00      6554
>    macro avg       0.50      0.50      0.50      6554
> weighted avg       0.99      1.00      0.99      6554
> 
> 
> 
> ... 
> 
> 
> 
> Category: direct_report
> Accuracy: 0.8937
>               precision    recall  f1-score   support
> 
>            0       0.90      0.97      0.94      5283
>            1       0.82      0.57      0.68      1271
> 
>     accuracy                           0.89      6554
>    macro avg       0.86      0.77      0.81      6554
> weighted avg       0.89      0.89      0.89      6554
> 
> 
> 
> Average of accuracy:0.9623959196128862
> ```







## Deploy

Using the **FLASK** framework, we can create a dashboard to visualize data and also deploy the machine learning model in the web backend.

You can use the `python run.py` command directly in the `/app` directory of this project, in a terminal, to run it locally.



First you will see some visualization of the cleaned data on the initial page.

<img src=".\src\visualization.png" style="zoom:70%;" />





Then enter a message in the top input box, and you can see the results of the classification:

<img src=".\src\classifier_result.png" style="zoom:70%;" />

The highlighted parts are the categories predicted by the model.


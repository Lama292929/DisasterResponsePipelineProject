import sys
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'wordnet'])
#le = preprocessing.LabelEncoder()
import warnings
from sqlalchemy import create_engine
import re
import pickle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing
warnings.simplefilter('ignore')
import subprocess
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('termcolor')
from termcolor import colored, cprint




def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # listing the columns
    categoryname = list(np.array(y.columns))

    return X, y, categoryname


def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    tokenss = word_tokenize(text)

    lemmatizerd = WordNetLemmatizer()

    clean_tokenss = []
    
    for tok in tokenss:

        clean_tok = lemmatizerd.lemmatize(tok).lower().strip()
        clean_tokenss.append(clean_tok)

    return clean_tokenss


def build_model():
     pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

     parameters = {'vect__min_df': [0.75, 0.50],
                   'tfidf__use_idf': [True, False],
                   'clf__estimator__n_estimators': [10, 25],
                   'clf__estimator__min_samples_split': [2, 4]}

     cv = GridSearchCV(pipeline, param_grid=parameters)


     return cv


def evaluate_model(model, X_test, Y_test, categoryname):
    
    y_pred = model.predict(X_test)

    for i, col in enumerate(categoryname):
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print('\nReport for the column ({}):\n'.format(colored(col, 'red', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'green')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'yellow')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'green')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'yellow')))

        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'green')))
        else:
            print('F-score: {}'.format(colored(round(fscore, 2), 'yellow')))



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categoryname = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categoryname)

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
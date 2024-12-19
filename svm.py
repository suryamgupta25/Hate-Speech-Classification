import pandas as pd
import emoji
import math
import numpy as np
import gensim.models.doc2vec as d2v
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
stemmer = SnowballStemmer('english')
stopset = set(stopwords.words('english'))
nltk.download('wordnet')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer, text
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time
from scipy.sparse import hstack, csr_matrix

MAX_SEQUENCE_LENGTH = 5000

# Function that converts a list of tokens into a feature dictionary, where features are represented as word:count pairs
def string_to_feature_dict(tokens):
    feature_dict = {}
    for word in tokens:
        if word not in feature_dict:
            feature_dict[word] = 1
        else:
            feature_dict[word] += 1
    return feature_dict

def get_advanced_features(model, dataset):
    """
    features = []
    for idx, row in dataset.iterrows():
        tokens = row['tweet']
        word_embeddings = []
        for word in tokens:
            if word in model:
                embedding = model[word]
                word_embeddings.append(embedding)
            else:
                word_embeddings.append(np.zeros(model.vector_size))
        
        if word_embeddings:
            word_embeddings = np.array(word_embeddings)
            avg_embedding = np.mean(word_embeddings, axis=0)
        else:
            avg_embedding = np.zeros(model.vector_size)
        
        features.append(avg_embedding)
    
    features_array = np.array(features)
    print("Features array shape:", features_array.shape)
    return features_array

    """
    features = []
    for idx, row in dataset.iterrows():
        word_embeddings = []
        for word in row['tweet']:
            if word in model:
                word_embeddings.extend(model[word])
            else:
                word_embeddings.extend([0] * 200)

        # Model normalizes to length MAX_SEQUENCE_LENGTH (first MAX_SEQUENCE_LENGTH features, or first MAX_SEQUENCE_LENGTH / 200 words)
        if len(word_embeddings) >= MAX_SEQUENCE_LENGTH:
            word_embeddings = word_embeddings[0:MAX_SEQUENCE_LENGTH]
        else:
            # Add filler values to ensure constant length feature matrix
            word_embeddings.extend([0] * (MAX_SEQUENCE_LENGTH - len(word_embeddings)))
        
        features.append(word_embeddings)

    return features

if __name__ == "__main__":
    df = pd.read_csv('labeled_data.csv', delimiter=',')
    df = pd.concat([df['class'].astype(str), df['tweet']], axis=1)

    lemmatizer = WordNetLemmatizer()

    # Split into train, validation, and test datasets
    # Want an equal amount of each class in each dataset
    class_distribution = {'0': 0, '1': 0, '2': 0}
    for idx, row in df.iterrows():
        s = row['tweet']
        newS = []

        # Convert emojis to text
        s = emoji.demojize(s)

        sTokens = s.split()
        # Remove any hyperlinks
        sTokens = [token for token in sTokens if not (token.lower().startswith('http://') or token.lower().startswith('https://'))]

        # Remove any mentions to people (tokens starting with @ or &
        sTokens = [token for token in sTokens if not (token.startswith("@") or token.startswith('&'))]

        # Remove any tokens that start with a series of ! (is used for indenting retweets in Reddit)
        for token in sTokens:
            re.sub(r'^!+', '', token.lower())

        # Lowercase all tokens and remove punctuation
        for token in sTokens:
            newS.append(re.sub(r'[^\w\s]','',token.lower()))

        # Remove tokens with numbers
        newS = ['' if re.search(r'.*[0-9]+.*', token) != None else token for token in newS]

        # Remove blank tokens after re.sub() commands, or if a token is equal to RT (retweet)
        newS = [t for t in newS if t != '' and t != "rt"]

        # Remove stopwords
        newS = [t for t in newS if t not in stopset]

        # Lemmatize words
        newS = [lemmatizer.lemmatize(t) for t in newS]
        
        row['tweet'] = newS
        if row['class'] not in class_distribution:
            class_distribution[row['class']] = 1
        else:
            class_distribution[row['class']] += 1

    print(df.head())
    
    print(class_distribution)

    # Take 1000 random samples from each class
    
    new_dataset = pd.DataFrame(columns=df.columns)
    for c in class_distribution:
        c_sample = df[(df['class'] == c)].sample(n=1000, random_state=42)
        new_dataset = pd.concat([new_dataset, c_sample],axis=0)
    # Training data is 70%, validation is 10%, test is 20%

    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    for c in class_distribution:
        c_sample = new_dataset[(new_dataset['class'] == c)]
        train_data, test_data = train_test_split(c_sample, test_size=0.2, random_state=42, stratify=c_sample['class'])

        train_df = pd.concat([train_df, train_data], axis=0)
        test_df = pd.concat([test_df, test_data], axis=0)

    """
    start_time = time.time()
    
    # Converts all lists of tokens in both datasets to word count dictionaries (features)
    train_features = train_df['tweet'].apply(string_to_feature_dict)
    test_features = test_df['tweet'].apply(string_to_feature_dict)

    # Vectorizes all feature dictionaries
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(train_features)
    X_test = vectorizer.transform(test_features)

    # Train and test a logistic regression model
    Y_train = train_df['class'].values
    Y_test = test_df['class'].values

    """
    start_time = time.time()

   # w2v = gensim.downloader.load('glove-twitter-200')
    

    
    # Get average GloVe embeddings for the text
   # X_train_emb = get_advanced_features(w2v, train_df)
   # X_test_emb = get_advanced_features(w2v, test_df)

   # scaler = StandardScaler()
   # X_train_emb_scaled = scaler.fit_transform(X_train_emb)
   # X_test_emb_scaled = scaler.transform(X_test_emb)
    
    
    # Add TF-IDF features
    # Prepare TF-IDF features
    train_texts = [' '.join(tokens) for tokens in train_df['tweet']]
    test_texts = [' '.join(tokens) for tokens in test_df['tweet']]

    vectorizer = text.TfidfVectorizer(max_features=2500)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    
    # Combine GloVe embeddings and TF-IDF features to make final feature matrix
    #X_train = hstack([X_train_tfidf, csr_matrix(X_train_emb_scaled)])
    #X_test= hstack([X_test_tfidf, csr_matrix(X_test_emb_scaled)])
    #X_train = X_train_emb
    #X_test = X_test_emb

    X_train = X_train_tfidf
    X_test = X_test_tfidf

    Y_train = train_df['class'].values
    Y_test = test_df['class'].values


    # Find best params by using the validation dataset for tuning hyperparameters
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
    }

    grid_search_model = GridSearchCV(LinearSVC(max_iter=50000), param_grid, cv=8, scoring='accuracy')
    grid_search_model.fit(X_train, Y_train)

    model = LinearSVC(**grid_search_model.best_params_)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    end_time = time.time()

    print(f"Elapsed time: {end_time-start_time:.2f} seconds")

    print(accuracy_score(Y_test, Y_pred))

    
    # Mapping features to model weights
    feature_names = vectorizer.get_feature_names_out()
    print(len(feature_names))
    weights = model.coef_.flatten()
    print(len(model.coef_))
    print(len(weights))


    weight_df = pd.DataFrame(columns=["name", "class", "weight"])
    for i in range(len(feature_names)):
        for j in range(3):
            weight_df.loc[len(weight_df)] = [feature_names[i], str(j), model.coef_[j][i]]

    weight_df = weight_df.sort_values(by='weight', ascending=False)

    # Reset index after sorting
    weight_df.reset_index(drop=True, inplace=True)

    print(weight_df.head())

    # Get top 5 words for each class
    important_words = {c: [] for c in class_distribution}
    for idx, row in weight_df.iterrows():
        if len(important_words[row['class']]) < 5:
            important_words[row['class']].append((row['name'], row['weight']))

            arraysFilled = True
            for c in class_distribution:
                if len(important_words[c]) < 5:
                    arraysFilled = False
            
            if arraysFilled:
                break

    print(weight_df.head())

    print("Top 5 most important words for each class: ")
    print(important_words)

    # Calculate results
    
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)
    print()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()


    prec0 = cm[0][0] / (cm[0][0] + cm[1][0] + cm[2][0])
    prec1 = cm[1][1] / (cm[0][1] + cm[1][1] + cm[2][1])
    prec2 = cm[2][2] / (cm[0][2] + cm[1][2] + cm[2][2])

    recall0 = cm[0][0] / (cm[0][0] + cm[0][1] + cm[0][2])
    recall1 = cm[1][1] / (cm[1][0] + cm[1][1] + cm[1][2])
    recall2 = cm[2][2] / (cm[2][0] + cm[2][1] + cm[2][2])

    print(f"Recall for class 0: {recall0:.2f}")
    print(f"Recall for class 1: {recall1:.2f}")
    print(f"Recall for class 2: {recall2:.2f}")
    print()
    print(f"Precision for class 0: {prec0:.2f}")
    print(f"Precision for class 1: {prec1:.2f}")
    print(f"Precision for class 2: {prec2:.2f}")
    print()
    print(f"F1 Score for class 0: {2*recall0*prec0 / (recall0 + prec0):.2f}")
    print(f"F1 Score for class 1: {2*recall1*prec1 / (recall1 + prec1):.2f}")
    print(f"F1 Score for class 2: {2*recall2*prec2 / (recall2 + prec2):.2f}")
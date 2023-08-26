from collections import defaultdict
import streamlit as st
import os


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
import pickle
from sklearn import metrics


# Define your constants
BASE_DIR = r'D:\Doc classifier\bbc'
LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']

# List of custom stopwords
custom_stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', "don't", 'Mr',
    'said', 'mr', 'used'
]

# Function to evaluate the classifier
def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    print(title + "Precision: %s" % precision)
    print(title + "Recall: %s" % recall)
    print(title + "F1: %s" % f1)


# Function to get the splits for training and testing
def get_splits(docs):
    random.shuffle(docs)

    X_train = []  # training documents
    y_train = []  # corresponding training labels

    X_test = []  # testing documents
    y_test = []  # corresponding testing labels

    pivot = int(0.8 * len(docs))

    for i in range(0, pivot):
        X_train.append(docs[i][1])
        y_train.append(docs[i][0])

    for i in range(pivot, len(docs)):
        X_test.append(docs[i][1])
        y_test.append(docs[i][0])

    return X_train, X_test, y_train, y_test

# Function to train the classifier
def train_classifier(docs):
    X_train, X_test, y_train, y_test = get_splits(docs)

    # the object that turns text into vectors of counts
    vectorizer = CountVectorizer(
        stop_words=custom_stop_words, ngram_range=(1, 3), min_df=3, analyzer='word'
    )

    # create document-term matrices
    dtm = vectorizer.fit_transform(X_train)

    # train Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)

    evaluate_classifier("Naive Bayes\tTrain\t", naive_bayes_classifier, vectorizer, X_train, y_train)
    evaluate_classifier("Naive Bayes\tTest\t", naive_bayes_classifier, vectorizer, X_test, y_test)

    # store the classifier and vectorizer in pickle files
    clf_filename = 'naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

    # store the vectorizer for transforming new documents to vectors in the future
    vect_filename = 'count_vectorizer.pkl'
    pickle.dump(vectorizer, open(vect_filename, 'wb'))
# Function to create the dataset
def create_data_set():
    with open('data.txt', 'w', encoding='utf8') as outfile:
        for label in LABELS:
            dir_path = os.path.join(BASE_DIR, label)
            for filename in os.listdir(dir_path):
                full_filename = os.path.join(dir_path, filename)
                print(full_filename)
                with open(full_filename, 'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n', ' ')
                    outfile.write("%s\t%s\t%s\n" % (label, filename, text))
# Function to setup documents for training
def setup_docs():
    docs = []  # list of tuples (label, text)
    with open('data.txt', 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split('\t')
            doc = (parts[0], parts[2].strip())
            docs.append(doc)
    return docs

# Function to classify text
def classify(text):
    #load classifier
    clf_filename = 'naive_bayes_classifier.pkl'
    nb_classifier = pickle.load(open(clf_filename, 'rb'))


    # load vectorizer
    vect_filename = 'count_vectorizer.pkl'
    vectorizer = pickle.load(open(vect_filename, 'rb'))

    pred = nb_classifier.predict(vectorizer.transform([text]))
    print(pred[0])


def classify_uploaded_text(text):
    # Load classifier
    clf_filename = 'naive_bayes_classifier.pkl'
    nb_classifier = pickle.load(open(clf_filename, 'rb'))

    # Load vectorizer
    vect_filename = 'count_vectorizer.pkl'
    vectorizer = pickle.load(open(vect_filename, 'rb'))

    pred = nb_classifier.predict(vectorizer.transform([text]))
    return pred[0]

# Streamlit UI
<<<<<<< HEAD
def main():
       # Add custom CSS styling
    st.markdown("""
        <style>
        .reportview-container {
            background-color: #e5e5f7;
            opacity: 0.8;
            background-image: radial-gradient(circle at center center, #000000, #e5e5f7),
                              repeating-radial-gradient(circle at center center, #000000, #000000, 10px, transparent 20px, transparent 10px);
            background-blend-mode: multiply;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("News Classifier")
    st.write("Upload a .txt or .pdf file for classification")
=======
>>>>>>> refs/remotes/origin/main

st.title("News Classifier")
st.write("Upload a .txt file or enter text for classification")

# Add a file uploader for .txt files
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

# Add a text input field for user input
user_input = st.text_area("Enter text here:", "")

# Check if a .txt file has been uploaded
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
        st.write("Uploaded text:")
        st.write(text)
        prediction = classify_uploaded_text(text)
        st.success("Prediction: {}".format(prediction))

# Check if the user has entered text
if user_input:
    prediction = classify(user_input)
    st.success("Prediction: {}".format(prediction))
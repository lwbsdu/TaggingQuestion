import sys
import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("../dataset/")

"""
merge body and tag

"""


def merge_tag_question():
    csv_question = pd.read_csv("../dataset/test_clean_question.csv")
    csv_tag = pd.read_csv("../dataset/testTag.csv")
    merge_tag = csv_tag.groupby("Id").aggregate(lambda x: list(x))
    merge_csv = pd.merge(csv_question, merge_tag, on='Id')
    return merge_csv


if os.path.exists("ovr.pkl"):
    print("ovr.pkl exist,begin to load!")
    load_ovr = joblib.load("ovr.pkl")
    exit()

data = merge_tag_question()
data_train = data["Body"]
data_tag = data["Tag"]

tf = TfidfVectorizer()
tf_data = tf.fit_transform(data_train)

mlb = MultiLabelBinarizer()
mlb_tag = mlb.fit_transform(data_tag)

x_train, x_test, y_train, y_test = train_test_split(tf_data, mlb_tag, test_size=0.1, random_state=31)

ovr = OneVsRestClassifier(SVC(gamma="auto"))
ovr.fit(x_train, y_train)

joblib.dump(ovr, 'ovr.pkl')



import csv
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

"""
pretreatment data, remove unnecessary text like
<pre><code> xx</code></pre>
and
<a href xxx>

but i think >xx</a> maybe useful

"""


def pretreatment_text(text):
    with_out_code = re.sub(r'<pre><code>(.|\n)*?</code></pre>', "", text)
    witch_out_href = re.sub(r'<a href=.*>', "", with_out_code)
    return witch_out_href


"""

 filter stopwords
 add  unnecessary words to stopwords

"""


def filter_stop_words(text):
    pre_text = pretreatment_text(text)
    tokenizer = RegexpTokenizer(r'\w+')
    text_tokenizer = tokenizer.tokenize(pre_text)
    stop_words = set(stopwords.words('english'))
    stop_words.add("p")
    stop_words.add("I")
    words = [stem_words(w) for w in text_tokenizer if w not in stop_words]
    print(words)
    return words


"""
 stem words
 use WordNetLemmatizer

"""


def stem_words(word):
    word_net = WordNetLemmatizer()
    return word_net.lemmatize(word)


csv_file = csv.reader(open("testQ.csv", 'r'))

clean_file = open("clean_question.csv", 'w')
csv_writer = csv.writer(clean_file)
for cs in csv_file:
    tittle_words = filter_stop_words(cs[5])
    filtered_words = filter_stop_words(cs[6])
    clean_words = tittle_words + filtered_words
    csv_writer.writerow([cs[0], clean_words])


import csv
from nltk.corpus import stopwords

csv_file = csv.reader(open("testQ.csv",'r'))

stop_words = set(stopwords.words('english'))

for cs in csv_file:
  print(cs[5])

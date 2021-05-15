import sys
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import string

# preprocesses the given string
# converts to lower case, removes punctuation and stop words
# and returns it back

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "film", "movie", "story", "one", "character", "time", "movies"]

def clean_data(text):
  lower_case = text.lower()
  wh_special_char = lower_case.translate(str.maketrans('','',string.punctuation))
  cleaned_data = ""
  for word in wh_special_char.split():
    if word not in stop_words:
      cleaned_data += " " + word
  return cleaned_data









if len(sys.argv) != 3:
    print("Three arguments expected.")
    exit()
pkl_filename = sys.argv[1]
FILE_PATH = sys.argv[2]

with open(pkl_filename, "rb") as file:
  clf = pickle.load(file)

data = []
labels = []
for file in os.listdir(FILE_PATH):
    filename=os.path.join(FILE_PATH,file)
    f = open(filename, "r", encoding="latin-1")
    text = f.read()
    data.append(clean_data(text))
    labels.append(file[-5])
    f.close()


count = CountVectorizer()
bag = count.transform(data)
print(bag.toarray())
#print(count.vocabulary_)
#print(bag.toarray())

prediction = clf.predict(bag.toarray())
print("".join(prediction))

import sys
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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
              "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "film", "movie", "story", "one",
              "character", "time", "movies", "br", "39", "quot"]

stop_words2 = ["br", "quot", "39"]

def clean_data(text):
  lower_case = text.lower()
  wh_special_char = lower_case.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
  cleaned_data = ""
  for word in wh_special_char.split():
    if word not in stop_words2:
      cleaned_data += " " + lemmatizer.lemmatize(word)
  return cleaned_data




if len(sys.argv) != 3:
    print("Three arguments expected.")
    exit()
pkl_filename = sys.argv[1]
FILE_PATH = sys.argv[2]

with open(pkl_filename, "rb") as file:
  clf = pickle.load(file)

with open("count.pkl", "rb") as file:
  count = pickle.load(file)

data = []
labels = []
for file in os.listdir(FILE_PATH):
    filename=os.path.join(FILE_PATH,file)
    f = open(filename, "r", encoding="latin-1")
    text = f.read()
    data.append(clean_data(text))
    labels.append(file[-5])
    f.close()


#count = CountVectorizer()
bag = count.transform(data)
print(bag.toarray())
#print(count.vocabulary_)
#print(bag.toarray())

prediction = clf.predict(bag.toarray())
print("".join(prediction))



count = 0
count_n = 0
count_p = 0
count_z = 0
label_n = 0
label_p = 0
label_z = 0
tp = 0
tn = 0
tz = 0
N = len(labels)
for i in range(N):
  if prediction[i] == 'P':
    count_p += 1
    if labels[i] == prediction[i]:
      tp += 1
  elif prediction[i] == 'N':
    count_n += 1
    if labels[i] == prediction[i]:
      tn += 1
  elif prediction[i] == 'Z':
    count_z += 1
    if labels[i] == prediction[i]:
      tz += 1

  if labels[i] == 'P':
    label_p += 1
  elif labels[i] == 'N':
    label_n += 1
  elif labels[i] == 'Z':
    label_z += 1

accuracy = (tp + tz + tn)*100/N
print("Train accuracy: " + str(accuracy))
  
precision_p = tp / count_p
precision_n = tn / count_n
precision_z = tz / count_z

recall_p = tp / label_p
recall_n = tn / label_n
recall_z = tz / label_z

print("Precisions (positive, negative, neutral): ")
print(precision_p)
print(precision_n)
print(precision_z)

print("Recalls (positive, negative, neutral): ")
print(recall_p)
print(recall_n)
print(recall_z)

print("Macro average accuracy: ")
print((recall_p + recall_n + recall_z) / 3)

print("Macro average precision: ")
print((precision_p + precision_n + precision_z) / 3)

print("Macro average recall: ")
print((recall_p + recall_n + recall_z) / 3)

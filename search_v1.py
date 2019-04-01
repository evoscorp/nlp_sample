import pandas as pd

# read json into a dataframe
df_idf=pd.read_json("data/stackoverflow-data-idf.json",lines=True)

# print schema
#print("Schema:\n\n",df_idf.dtypes)
#print("Number of questions,columns=",df_idf.shape)


import re
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

df_idf['text'] = df_idf['title'] + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))

#show the first 'text'
#print(df_idf['text'][2])



from sklearn.feature_extraction.text import CountVectorizer
import re

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

#load a set of stop words
stopwords=get_stop_words("resources/stopwords.txt")

#get the text column 
docs=df_idf['text'].tolist()
original_doc = df_idf['text'].tolist()
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

while True:
    search = input("Input query: ")
    phrase = [search]


    # You could do it like this:
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    all_phrases = phrase + docs
    my_features = vectorizer.fit_transform(all_phrases)
    scores = (my_features[0, :] * my_features[1:, :].T).A[0]
    #print(scores[0])
    a = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1] ,reverse = True)]
    rslt = a[:10]
    #best_score = np.argmax(scores)
    for i in range(len(rslt)):
        print("Rank {0} - Line[{1}] : \n\t {2}\n".format(i + 1 ,rslt[i] + 1, original_doc[int(rslt[i])]))
        
    #answer = original_doc[best_score]

    #print("\n\n\n ANSWER :: {0} \n".format(best_score))
    #print(answer)

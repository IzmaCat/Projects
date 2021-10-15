
# -*- coding: utf-8 -*-


import glob
import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk import tokenize
import spacy
import matplotlib.pyplot as plt 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils

from tqdm import tqdm
import multiprocessing
import nltk
import pandas as pd
#tqdm.pandas(desc="progress-bar")
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



df=pd.read_csv("All_jobs.csv")
my_hard_list=pd.read_excel("HARD_SKILLS_LIST.xlsx")
my_soft_list=pd.read_excel("SOFT_SKILLS_LIST.xlsx")





def cleanText(text):
   
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


df['Description_after_keywords']=df['Description_after_keywords'].apply(str)
df['Description_after_keywords'] = df['Description_after_keywords'].apply(cleanText)

from textblob import Word
## lemmatization
df['Description_after_keywords']  = df['Description_after_keywords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


## Lower case
df['Description_after_keywords'] = df['Description_after_keywords'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## remove tabulation and punctuation
#df['Description_after_keywords']  = df['Description_after_keywords'].str.replace('[^\w\s]',' ')
## digits
df['Description_after_keywords']  =df['Description_after_keywords'].str.replace('\d+', '')

#remove stop words

df['Description_after_keywords']  = df['Description_after_keywords'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))




#split the text into sentences 

sentences=[]
for text in df.Description_after_keywords:
    sentences.append(tokenize.sent_tokenize(text))


flat_sent = [item for sublist in sentences for item in sublist]


df_sent = pd.DataFrame(flat_sent,columns=["sentences"])

spacy_tok = spacy.load('en_core_web_sm')

#the below operation takes time
#please open tokenized_text.csv

p=[]
#extracting tokens lemmas and pos tags 
for row in df_sent.sentences:
    parsed = spacy_tok(row)
    p.append(parsed)

tokens=[]
lemmas=[]
pos=[]
tags=[]

for ele in p:
    
    for i, token in enumerate(ele):
        tokens.append(token.text)
        lemmas.append(token.lemma_)
        pos.append(token.pos_)
        tags.append(token.tag_)
        
    
tokenized_text = pd.DataFrame()

tokenized_text['text'] = tokens
tokenized_text['lemma'] = lemmas
tokenized_text['pos'] = pos
tokenized_text['tag'] = tags


tokenized_text=tokenized_text.drop_duplicates(subset="text",keep="first")
#tokenized_text.to_csv("tokenized_text.csv")
#FIND SIMILARITY SCORE BETWEEN TOKENS AND WORDS FROM THE SKILLS LISTS
def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


list_tokens = tokenized_text.text.tolist()
list_hard=my_hard_list['Skills'].tolist()
sim=[]
#threshold = 0.80     # if needed
for key in list_tokens:
    for word in list_hard:
        try:
            # print(key)
            # print(word)
            res = cosdis(word2vec(word), word2vec(key))
            
            # print(res)
          #  print("The cosine similarity between : {} and : {} is: {}".format(word, key, res*100))
            
            #if res > threshold:
            sim.append((key, word, res))
           
        except IndexError:
            pass



sim_df_hard=pd.DataFrame(sim, columns=('token', 'match', 'similarity_score'))

#sim_df_hard=pd.read_csv("sim.csv")

sim_df_hard
sim_df_hard.sort_values('similarity_score').drop_duplicates(subset=['token', 'match'], keep='last')
sim_df_hard.loc[ sim_df_hard['similarity_score'] >= 0.80,"type" ]="Hard_skill"#anything above this threshold tagged as hard skill
sim_df_hard.type.fillna('not_skill', inplace=True)
sim_df_hard.reset_index(drop=True, inplace=True)
sim_df_hard


list_soft=my_soft_list["Skills"].tolist()
sim_s=[]
#threshold = 0.80     # if needed
for key in list_tokens:
    for word in list_soft:
        try:
            # print(key)
            # print(word)
            res = cosdis(word2vec(word), word2vec(key))
            
            # print(res)
            #print("The cosine similarity between : {} and : {} is: {}".format(word, key, res*100))
            
            #if res > threshold:
            sim_s.append((key, word, res))
           
        except IndexError:
            pass

sim_df_soft=pd.DataFrame(sim_s, columns=('token', 'match', 'similarity_score'))


#sim_df_soft.to_csv("sim_soft.csv")

sim_df_soft
sim_df_soft.sort_values('similarity_score').drop_duplicates(subset=['token', 'match'], keep='last')
sim_df_soft.loc[ sim_df_soft['similarity_score'] >= 0.80,"type" ]="Soft_skill" #anything above this threshold tagged as hard skill
sim_df_soft.type.fillna('not_skill', inplace=True)
sim_df_soft.reset_index(drop=True, inplace=True)
sim_df_soft



#create a df with the tagged tokens 
frames=[sim_df_hard,sim_df_soft]
df_final=pd.concat(frames)
df_final.reset_index(drop=True, inplace=True)
#remove duplicates
df_final=df_final.sort_values('similarity_score').drop_duplicates(subset=['token', 'match'], keep='last')


#Checking balance
plt.figure()
pd.value_counts(df_final['type']).plot.bar(title="Type of skill distribution in df")
plt.xlabel("type")
plt.ylabel("No. of rows in df")
plt.show()

'''not_skill     8606834
Soft_skill      57976
Hard_skill      37965'''



#the dataset is very unbalnced so we are going to create a smaller balnced dataset


from sklearn.feature_extraction.text import  TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



df_final.type.replace('not_skill',0,inplace=True)
df_final.type.replace("Hard_skill",1,inplace=True)
df_final.type.replace("Soft_skill",2,inplace=True)

def get_top_data(top_n = 50000):
    top_noskill = df_final[df_final['type'] == 0].head(top_n)
    top_hard = df_final[df_final['type'] == 1].head(top_n)
    top_soft = df_final[df_final['type'] ==2].head(top_n)
    top_df_small = pd.concat([top_hard,top_soft,top_noskill])
    return top_df_small

# Function call to get the top 50000 from each type
top_df_small = get_top_data(top_n=50000)
top_df_small.reset_index(drop=True, inplace=True)
# After selecting top few samples of each sentiment
print("After segregating and taking equal number of rows for each type:")
print(top_df_small['type'].value_counts())
top_df_small.head(100)

## Converting text to features 
vectorizer = TfidfVectorizer()
#Tokenize and build vocabulary
X = vectorizer.fit_transform(top_df_small.token) 
y = top_df_small.type
#X = vectorizer.fit_transform(df_final.token) 
#y=df_final.type


# split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
print("train data shape: ",X_train.shape)
print("test data shape: ",X_test.shape)
#Let's do a quick sanity check for the distribution of our train and test data.

y_train.hist()
y_test.hist()

# Fit model Naive Bayes
clf = MultinomialNB()
clf=clf.fit(X_train, y_train)
## Predict
y_predictedCLF = clf.predict(X_test)

#Fit model Logistic regression
# Fit model
lg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lg=lg.fit(X_train, y_train)
## Predict
y_predictedLG = lg.predict(X_test)




#MODEL EVALUATION
#evaluate the predictions
print("Accuracy score  NB is: ",accuracy_score(y_test, y_predictedCLF)) #  
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedCLF))
print("Classification Report NB: ")
print(classification_report(y_test, y_predictedCLF))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_test, y_predictedCLF)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
plt.show()
    

print("Accuracy score  LG is: ",accuracy_score(y_test, y_predictedLG)) #  
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedLG))
print("Classification Report LG: ")
print(classification_report(y_test, y_predictedLG))














####Doc2vec approach
train, test = train_test_split(top_df_small, test_size=0.3, random_state=42)


train_tagged = train.apply(
    lambda r: TaggedDocument(words=[r.token], tags=[r.type]), axis=1) #tagging documents per level
test_tagged = test.apply(
    lambda r: TaggedDocument(words=[r.token], tags=[r.type]), axis=1)

train_tagged.values[2]




model_dbow1 = Doc2Vec(dm=1, vector_size=300, negative=5, min_count=1,window=8 ) #Distributed memory algorithm
model_dbow=Doc2Vec( dm=0,vector_size=300, negative=5, min_count=1, alpha=0.0254,window=8) #Distributed Bag of words algorithm
model_dbow1.build_vocab([x for x in tqdm(train_tagged.values)])
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])



for epoch in range(10):
    model_dbow1.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow1.alpha -= 0.002
    model_dbow1.min_alpha = model_dbow.alpha
    

for epoch in range(10):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
    
        
    
    
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

y_train1, X_train1 = vec_for_learning(model_dbow1, train_tagged)
y_test1, X_test1 = vec_for_learning(model_dbow1, test_tagged)
logreg1 = LogisticRegression(n_jobs=1, C=1e5)
logreg1.fit(X_train1, y_train1)
y_pred1 = logreg1.predict(X_test1)




print('Testing accuracy from DM algorith %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score DM algorith: {}'.format(f1_score(y_test, y_pred, average='weighted')))

print('Testing accuracy DBow algorith %s' % accuracy_score(y_test1, y_pred1))
print('Testing F1 score  DBow algorith: {}'.format(f1_score(y_test1, y_pred1, average='weighted')))







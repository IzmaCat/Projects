
# -*- coding: utf-8 -*-

import glob
import numpy as np
import pandas as pd
import string
import re
import nltk
import glob
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from nltk.tokenize import WordPunctTokenizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('wordnet')
en_stops = set(stopwords.words('english'))
from nltk import FreqDist



'''
Open every file  in the folder,this method was needed because the webscrapping generated a csv file each time 
and we can add any new file any time


#Create a dataframe for each of the 3 senority levels
jobs_entry_level = []
for infile in glob.glob("C:/Users/*/OneDrive/Desktop/Jobs_round 2/entry/*csv"):
    data = pd.read_csv(infile)
   
    jobs_entry_level.append(data)

entry_level = pd.concat(jobs_entry_level).reset_index(drop=True)
entry_level = entry_level.replace(to_replace='None', value=np.nan).dropna()
entry_level.drop_duplicates(subset ="Description", keep = "first",inplace=True)
entry_level.reset_index(drop=True, inplace=True)



jobs_mid_level = []
for infile in glob.glob("C:/Users/*/OneDrive/Desktop/Jobs_round 2/mid/*csv"):
    data = pd.read_csv(infile)
   
    jobs_mid_level.append(data)

mid_level = pd.concat(jobs_mid_level).reset_index(drop=True)
mid_level = mid_level.replace(to_replace='None', value=np.nan).dropna()
mid_level.drop_duplicates(subset ="Description", keep = "first",inplace=True)
mid_level.reset_index(drop=True, inplace=True)
#mid_level.to_csv('jobs_mid_level.csv')



jobs_senior_level = []
for infile in glob.glob("C:/Users/*/OneDrive/Desktop/Jobs_round 2/senior/*csv"):
    data = pd.read_csv(infile)
   
    jobs_senior_level.append(data)

senior_level = pd.concat(jobs_senior_level).reset_index(drop=True)
senior_level = senior_level.replace(to_replace='None', value=np.nan).dropna()
senior_level.drop_duplicates(subset ="Description", keep = "first",inplace=True)
senior_level.reset_index(drop=True, inplace=True)

#number of job posts per senority level
print("Entry level jobs",entry_level.shape[0])
print("Mid level jobs",mid_level.shape[0])
print("Senior level jobs",senior_level.shape[0])
'''


entry_level=pd.read_csv("ENTRY.csv")
mid_level=pd.read_csv("MID.csv")
senior_level=pd.read_csv("SENIOR.csv")



'''
At this point I am interested in the job Titles of the job Posts
Keywords in job titles will help understand the hiererchy in senority'''


#Let's check the unique job titles for every level
title_e=entry_level.Title.unique().tolist()
title_m=mid_level.Title.unique().tolist()
title_s= senior_level.Title.unique().tolist()

print("Unique titles in entry level:",len(title_e))
print("Unique titles in mid level:",len(title_m))
print("Unique titles in senior level:",len(title_s))

#We can detect that job titles are not yet established and there is a huge variety concerning the diversity inclusion & enquity


title_counts_e = entry_level.Title.value_counts()
title_counts_m = mid_level.Title.value_counts()
title_counts_s = senior_level.Title.value_counts()

print("Entry level Job titles counts \n",title_counts_e) #Titles such as Business Analyst and Data Analyst are frequent but 12 and 8 times appear/not enough for conclusions
print("\n")
print("Mid level Job titles counts \n",title_counts_m)#Diversity & Inclusion Specialist 7 times
print("\n")
print("Senior level Job titles counts \n",title_counts_s)#Director of Human Resources      5 times



'''Since I cannot draw any conclusions from the unique job titles 
I will analyse the keywords;I will find frequent words or unigrams that show o proper job title and proof of senority
for example I would expect to entry level to see interniships,trainee etc,in mid level associate,manager etc and in senior level director senior manager etc.
Of course I am more interested in the word that accompany these words of seniority hierarchy.
'''

'''The function below will clean the titles create tokens of words remove punctuation ,stopwords and numbers and convert to lower case '''


#cleaning

def clean_parts(part):
   
    # remove hyperlinks
    part = re.sub(r'https?:\/\/.*[\r\n]*', '', part)
    
    # remove hashtags etc
    # only removing the hash # sign from the word and numbers
    part = re.sub(r'#', '’',part)
    part = re.sub(r'’', '', part)
    part = re.sub(r'”', '', part)
    part=re.sub(r'\d+', '', part)
    # tokenize job descriptions
    tokenizer = WordPunctTokenizer()
    part_jobs = tokenizer.tokenize(part)
 
    part_clean = []    
    for word in part_jobs:
     if word not in en_stops and word not in string.punctuation:
    
      word=word.lower() #make every word lower case
      #lem_word = wnl.lemmatize( word) # lemmatizing word
              
      part_clean.append(word )
    return part_clean              

a=[]
for title in entry_level.Title:
    a.append(clean_parts(title))

entry_level["tokenized_Title"]=a
entry_level["tokenized_Title"]=[" ".join(part) for part in entry_level["tokenized_Title"].values]

################ENTRY LEVEL######


from sklearn.feature_extraction.text import CountVectorizer

#FREQUENT WORDS 
###ENTRY_LEVEL
c_vec = CountVectorizer(stop_words=en_stops, ngram_range=(1,1))
# matrix of ngrams
ngrams_entry = c_vec.fit_transform(entry_level["tokenized_Title"])
# count frequency of ngrams
count_values_entry = ngrams_entry.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram_entry_level = pd.DataFrame(sorted([(count_values_entry[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words_in_Titles'})


print(df_ngram_entry_level.head(50))


df_ngram_entry_level['Job Seniority Level']="Entry Level"

#df_ngram_entry_level.to_csv('entry_level.csv')

###MID_LEVEL


b=[]
for title in mid_level.Title:
    b.append(clean_parts(title))

mid_level["tokenized_Title"]=b
mid_level["tokenized_Title"]=[" ".join(part) for part in mid_level["tokenized_Title"].values]


# matrix of ngrams
ngrams_mid = c_vec.fit_transform(mid_level["tokenized_Title"])
# count frequency of ngrams
count_values_mid = ngrams_mid.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram_mid_level = pd.DataFrame(sorted([(count_values_mid[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words_in_Titles'})


print(df_ngram_mid_level.head(50))


df_ngram_mid_level['Job Seniority Level']="Mid Level"

#df_ngram_mid_level.to_csv('mid_level.csv')


###SENIOR_LEVEL


c=[]
for title in senior_level.Title:
    c.append(clean_parts(title))

senior_level["tokenized_Title"]=c
senior_level["tokenized_Title"]=[" ".join(part) for part in senior_level["tokenized_Title"].values]


ngrams_sen = c_vec.fit_transform(senior_level["tokenized_Title"])
# count frequency of ngrams
count_values_sen = ngrams_sen.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram_sen_level = pd.DataFrame(sorted([(count_values_sen[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words_in_Titles'})


print(df_ngram_sen_level.head(50))


df_ngram_sen_level['Job Seniority Level']="Senior Level"

#df_ngram_sen_level.to_csv('senior_level.csv')

###create one dataframe for all levels
frames=[df_ngram_entry_level,df_ngram_mid_level,df_ngram_sen_level]

dfs=pd.concat(frames)
dfs.reset_index(drop=True, inplace=True)

dfs.to_csv("frequent_words_titles.csv")



#####Unigrams#####withoutapplying tokenizations on titles
###ENTRY_LEVEL

c_vec = CountVectorizer(stop_words=en_stops, ngram_range=(2,3))
# matrix of ngrams
ngrams_entry = c_vec.fit_transform(entry_level["Title"])
# count frequency of ngrams
count_values_entry = ngrams_entry.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram_entry_level = pd.DataFrame(sorted([(count_values_entry[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words_in_Titles'})


print(df_ngram_entry_level.head(50))


df_ngram_entry_level['Job Seniority Level']="Entry Level"


###MID_LEVEL
# matrix of ngrams
ngrams_mid = c_vec.fit_transform(mid_level['Title'])
# count frequency of ngrams
count_values_mid = ngrams_mid.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram_mid_level = pd.DataFrame(sorted([(count_values_mid[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words_in_Titles'})


print(df_ngram_mid_level.head(50))


df_ngram_mid_level['Job Seniority Level']="Mid Level"

###SENIOR_LEVEL
ngrams_sen = c_vec.fit_transform(senior_level['Title'])
# count frequency of ngrams
count_values_sen = ngrams_sen.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram_sen_level = pd.DataFrame(sorted([(count_values_sen[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words_in_Titles'})


print(df_ngram_sen_level.head(50))


df_ngram_sen_level['Job Seniority Level']="Senior Level"

#create one dataframe for unigrams
frames=[df_ngram_entry_level,df_ngram_mid_level,df_ngram_sen_level]

dfs_unigrams=pd.concat(frames)
dfs_unigrams.reset_index(drop=True, inplace=True)

dfs_unigrams.to_csv("Unigrams.csv")


''' link to visualization : 
    
   https://public.tableau.com/views/FrequentwordsinJobTitles/Dashboard2?:language=en-GB&:display_count=n&:origin=viz_share_link

'''

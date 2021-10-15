
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('wordnet')
stop = stopwords.words('english')
from plotly.offline import plot
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 
from textblob import Word
from sklearn import tree
from sklearn.neural_network import MLPClassifier


df_entry=pd.read_csv("ENTRY_new.csv")
df_mid=pd.read_csv("MID.csv")
df_senior=pd.read_csv("SENIOR_new.csv")




'''
Create a funtion that "cleans" our text 
-remove hyperlinks,symbols,numbers,punctuation,stopwords


'''

def cleanText(text):
   
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('$', '')
    ## digits
    ## remove tabulation and punctuation
    return text
    


df_entry['Description_after_keywords']=df_entry['Description_after_keywords'].apply(str)
df_entry['Description_after_keywords'] = df_entry['Description_after_keywords'].apply(cleanText)
## Lower case
df_entry['Description_CLEAN'] = df_entry['Description_after_keywords'].apply(lambda x: " ".join(x.lower()for x in x.split()))
df_entry['Description_CLEAN']  = df_entry['Description_after_keywords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
## stopwords
df_entry['Description_CLEAN']  = df_entry['Description_after_keywords'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df_entry['Description_CLEAN']  =df_entry['Description_after_keywords'].str.replace('\d+', '')
df_entry['Description_CLEAN']  = df_entry['Description_after_keywords'].str.replace('[^\w\s]',' ')

df_mid['Description_after_keywords']=df_mid['Description_after_keywords'].apply(str)
df_mid['Description_after_keywords'] = df_mid['Description_after_keywords'].apply(cleanText)

## Lower case
df_mid['Description_CLEAN'] = df_mid['Description_after_keywords'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## stopwords
df_mid['Description_CLEAN']  = df_mid['Description_after_keywords'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df_mid['Description_CLEAN']  = df_mid['Description_after_keywords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df_mid['Description_CLEAN']  =df_mid['Description_after_keywords'].str.replace('\d+', '')
df_mid['Description_CLEAN']  = df_mid['Description_after_keywords'].str.replace('[^\w\s]',' ')

df_senior['Description_after_keywords']=df_senior['Description_after_keywords'].apply(str)
df_senior['Description_CLEAN'] = df_senior['Description_after_keywords'].apply(cleanText)

## Lower case
df_senior['Description_CLEAN'] = df_senior['Description_after_keywords'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## digits
df_senior['Description_CLEAN']  = df_senior['Description_after_keywords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df_senior['Description_CLEAN']  = df_senior['Description_after_keywords'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df_senior['Description_CLEAN']  =df_senior['Description_after_keywords'].str.replace('\d+', '')
df_senior['Description_CLEAN']  = df_senior['Description_after_keywords'].str.replace('[^\w\s]',' ')#remove tabulation and punctuation


from sklearn.feature_extraction.text import CountVectorizer


def tokenize(text):
    #tokens=[]
    tokenizer = WordPunctTokenizer()
    token= tokenizer.tokenize(text)
    #tokens.append(token)
    return token


df_entry['Tokenized']=df_entry['Description_CLEAN'].apply(tokenize)
df_mid['Tokenized']=df_mid['Description_CLEAN'].apply(tokenize)
df_senior['Tokenized']=df_senior['Description_CLEAN'].apply(tokenize)


#entry level
df_entry['Tokenized_values']=[" ".join(part) for part in df_entry['Tokenized'].values]

df_entry["Level"]="Entry"

c_vec = CountVectorizer(stop_words=stop, ngram_range=(2,3))
# matrix of ngrams
ngrams_entry_desc = c_vec.fit_transform(df_entry["Tokenized_values"])
# count frequency of ngrams
count_values_entry_desc = ngrams_entry_desc.toarray().sum(axis=0)
# list of ngrams
vocab_desc = c_vec.vocabulary_
df_ngram_entry = pd.DataFrame(sorted([(count_values_entry_desc[i],k) for k,i in vocab_desc.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words'})

df_ngram_entry["Level"]="Entry"


#keep high frequency word/ to reduce size
entry_most_com=df_ngram_entry[df_ngram_entry['frequency']>100]



#midlevel
df_mid['Tokenized_values']=[" ".join(part) for part in df_mid['Tokenized'].values]

df_mid['Level']='Mid'
 #MID LEVEL 
c_vec = CountVectorizer(stop_words=stop, ngram_range=(2,3)) #find bigrams and trigrams
ngrams_mid_desc = c_vec.fit_transform(df_mid["Tokenized_values"])
# count frequency of ngrams
count_values_mid_desc = ngrams_mid_desc.toarray().sum(axis=0)
# list of ngrams
vocab_desc_mid = c_vec.vocabulary_
df_ngram_mid = pd.DataFrame(sorted([(count_values_mid_desc[i],k) for k,i in vocab_desc_mid.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words'})


df_ngram_mid["Level"]="Mid"
#keep high frequency word/ to reduce size
mid_most_com=df_ngram_mid[df_ngram_mid['frequency']>100]



###SENIOR_LEVEL


df_senior['Tokenized_values']=[" ".join(part) for part in df_senior['Tokenized'].values]

df_senior['Level']='Senior'



c_vec = CountVectorizer(stop_words=stop, ngram_range=(1,2))
# matrix of ngrams
ngrams_senior_desc = c_vec.fit_transform(df_senior["Tokenized_values"])
# count frequency of ngrams
count_values_senior_desc = ngrams_entry_desc.toarray().sum(axis=0)
# list of ngrams
vocab_desc_senior = c_vec.vocabulary_
df_ngram_senior = pd.DataFrame(sorted([(count_values_senior_desc[i],k) for k,i in vocab_desc_senior.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'Job_Frequent_Words'})

df_ngram_senior['Level']='Senior'


#keep high frequency word/ to reduce size
senior_most_com=df_ngram_senior[df_ngram_senior['frequency']>100]

##Create dataframe for all levels and descriptions
frames_all=[df_entry,df_mid,df_senior]


dfs_text=pd.concat(frames_all)#raw text
dfs_text.reset_index(drop=True, inplace=True)

dfs_text.Level.replace("Entry",0,inplace=True)
dfs_text.Level.replace("Mid",1,inplace=True)
dfs_text.Level.replace("Senior",2,inplace=True)

dfs_text.head()
#dfs_text.to_csv('Jobs_by_Senority_Level.csv')

frames_ngrams_desc=[df_ngram_entry,df_ngram_mid,df_ngram_senior]
dfs=pd.concat(frames_ngrams_desc)#unigrams
dfs.reset_index(drop=True, inplace=True)
#dfs.to_csv("unigrams.csv")
frames_high_frequency=[senior_most_com,mid_most_com,entry_most_com]
com_levels=pd.concat(frames_high_frequency)
com_levels.reset_index(drop=True, inplace=True)

###SENIOR_LEVEL
df_senior['Tokenized_values']=[" ".join(part) for part in df_senior['Tokenized'].values]

df_senior['Level']='Senior'

frames_all=[df_entry,df_mid,df_senior]

dfs_text=pd.concat(frames_all)#raw text
dfs_text.reset_index(drop=True, inplace=True)

dfs_text.Level.replace("Entry",0,inplace=True)
dfs_text.Level.replace("Mid",1,inplace=True)
dfs_text.Level.replace("Senior",2,inplace=True)

dfs_text.head()
#dfs_text.to_csv('Jobs_by_Senority_Level.csv')

#com_levels.to_csv("Com_levels.csv")

#A quick visualization of 20 top Unigrams per seniority level
e10=entry_most_com[:10]
m10=mid_most_com[:10]
s10=senior_most_com[:10]
a=[e10,m10,s10]
top=pd.concat(a)
top.reset_index(drop=True, inplace=True)

fig = px.bar(top, x="Job_Frequent_Words", y="frequency",text="frequency",color = "Level" ,title="TOP 10 Unigrams In Job Descriptions per seniority level")
#fig.update_traces(marker_color=px.colors.qualitative.Pastel)
fig.update_layout(barmode='stack')
fig.update_xaxes(categoryorder='total ascending')
fig.show()
plot(fig, filename="TOP 10 Unigrams In Job Descriptions per seniority level.html")



#MODELING
    
'''
Modeling
We are now going to translate this skill-extraction problem into a classification one first. And then extract the most important features from each class.

The most important features, in this case, represent the words that most likely will belong to a class ( in our case senority level)

'''
#checking the balance of the data for unigrams
plt.figure()
pd.value_counts(dfs['Level']).plot.bar(title="Type of skill distribution in df")
plt.xlabel("type")
plt.ylabel("No. of rows in df")
plt.show()
dfs.Level.replace('Entry',0,inplace=True)
dfs.Level.replace("Mid",1,inplace=True)
dfs.Level.replace("Senior",2,inplace=True)
##1rst Approach Raw text

## Converting text to features 
vectorizer = TfidfVectorizer()
#Tokenize and build vocabulary
X = vectorizer.fit_transform(dfs_text.Description) 
y = dfs_text.Level

# split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
print("train data shape: ",X_train.shape)
print("test data shape: ",X_test.shape)
#Let's do a quick sanity check for the distribution of our train and test data.

y_train.hist()
y_test.hist()

# Fit model Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)
## Predict
y_predictedCLF = clf.predict(X_test)


#Fit model Logistic regression
# Fit model
lg2 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lg2.fit(X_train, y_train)
## Predict
y_predictedLG2 = lg2.predict(X_test)



clfDT =  tree.DecisionTreeClassifier(criterion='gini')

#Classifier training                 
clfDT.fit(X_train, y_train)

#  Test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)

# defining neural network 1


print("Accuracy score DT is: ",accuracy_score(y_test, Y_test_pred_DT)) #0.667

#MODEL EVALUATION
#evaluate the predictions
print("Accuracy score  NB is: ",accuracy_score(y_test, y_predictedCLF)) # 0.0.53
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedCLF))
print("Classification Report NB: ")
print(classification_report(y_test, y_predictedCLF))

print("Accuracy score  LG is: ",accuracy_score(y_test, y_predictedLG2)) #  0.704
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedLG2))
print("Classification Report LG: ")
print(classification_report(y_test, y_predictedLG2))

##2nd Approach Unigrams
## Converting text to features 
vectorizer = TfidfVectorizer()
#Tokenize and build vocabulary
X = vectorizer.fit_transform(dfs.Job_Frequent_Words) 
y = dfs.Level

# split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
print("train data shape: ",X_train.shape)
print("test data shape: ",X_test.shape)
#Let's do a quick sanity check for the distribution of our train and test data.

y_train.hist()
y_test.hist()

# Fit model Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)
## Predict
y_predictedCLF = clf.predict(X_test)

#Fit model Logistic regression Multi
# Fit model
lg1 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lg1.fit(X_train, y_train)
## Predict
y_predictedLG1 = lg1.predict(X_test)



#MODEL EVALUATION
#evaluate the predictions
print("Accuracy score  NB is: ",accuracy_score(y_test, y_predictedCLF)) # 0.625
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedCLF))
print("Classification Report NB: ")
print(classification_report(y_test, y_predictedCLF))

print("Accuracy score  LG is: ",accuracy_score(y_test, y_predictedLG1)) # 0.62
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedLG1))
print("Classification Report LG: ")
print(classification_report(y_test, y_predictedLG1))



##3rd approach Tokens 
dfs_text["Tokens"]=dfs_text["Tokenized"].astype(str)
dfs_text["Tokens"] =dfs_text["Tokens"].str.replace('[^\w\s]',' ')
dfs_text["Tokens"]=dfs_text["Tokens"].apply(lambda x: x.strip("[").strip("]"))

## Converting text to features 
vectorizer = TfidfVectorizer()
#Tokenize and build vocabulary
X = vectorizer.fit_transform(dfs_text.Tokens) 
y = dfs_text.Level

# split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
print("train data shape: ",X_train.shape)
print("test data shape: ",X_test.shape)
#Let's do a quick sanity check for the distribution of our train and test data.

y_train.hist()
y_test.hist()

# Fit model Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)
## Predict
y_predictedCLF = clf.predict(X_test)


#Fit model Logistic regression
# Fit model
lg2 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lg2.fit(X_train, y_train)
## Predict
y_predictedLG2 = lg2.predict(X_test)



clfDT =  tree.DecisionTreeClassifier(criterion='gini')

#Classifier training                 
clfDT.fit(X_train, y_train)

#  Test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)

# defining neural network 1


print("Accuracy score DT is: ",accuracy_score(y_test, Y_test_pred_DT)) #0.59

#MODEL EVALUATION
#evaluate the predictions
print("Accuracy score  NB is: ",accuracy_score(y_test, y_predictedCLF)) # 0.54
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedCLF))
print("Classification Report NB: ")
print(classification_report(y_test, y_predictedCLF))

print("Accuracy score  LG is: ",accuracy_score(y_test, y_predictedLG2)) #  0.68
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedLG2))
print("Classification Report LG: ")
print(classification_report(y_test, y_predictedLG2))


##4rth approach Clean Text

## Converting text to features 
vectorizer = TfidfVectorizer()
#Tokenize and build vocabulary
X = vectorizer.fit_transform(dfs_text.Description_CLEAN) 
y = dfs_text.Level

# split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) 
print("train data shape: ",X_train.shape)
print("test data shape: ",X_test.shape)
#Let's do a quick sanity check for the distribution of our train and test data.

y_train.hist()
y_test.hist()

# Fit model Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)
## Predict
y_predictedCLF = clf.predict(X_test)


#Fit model Logistic regression
# Fit model
lg2 = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lg2.fit(X_train, y_train)
## Predict
y_predictedLG2 = lg2.predict(X_test)



clfDT =  tree.DecisionTreeClassifier(criterion='gini')

#Classifier training                 
clfDT.fit(X_train, y_train)

#  Test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)




print("Accuracy score DT is: ",accuracy_score(y_test, Y_test_pred_DT)) #0.590

#MODEL EVALUATION
#evaluate the predictions
print("Accuracy score  NB is: ",accuracy_score(y_test, y_predictedCLF)) # 0.54
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedCLF))
print("Classification Report NB: ")
print(classification_report(y_test, y_predictedCLF))

print("Accuracy score  LG is: ",accuracy_score(y_test, y_predictedLG2)) #  0.687
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedLG2))
print("Classification Report LG: ")
print(classification_report(y_test, y_predictedLG2))







'''Let's now extract the most meaningful features of each class.

To do so, we can access the attribute feature_logprob from our model which returns the log probability of features given a class.

We will next sort the log probabilies descendingly.

And finally map the most important tokens to the classes
'''

# we have for each class/level a list of the most representative words/tokens found in job descriptions.



''' shrink this list of words to only 20
    we will use the skills from the manually collected list as HARD SKILLS 
        and the library TextBlob to identify adjectives which will represent the soft skills'''



from textblob import TextBlob
#from nltk import pos_tag  
#from rake_nltk import Rake
#rake_nltk_var = Rake()

my_hard_list=pd.read_excel("HARD_SKILLS_LIST.xlsx")
my_soft_list=pd.read_excel("SOFT_SKILLS_LIST.xlsx")
hard_s=my_hard_list['Skills'].tolist()

soft_s=my_soft_list["Skills"].tolist()

soft_s= list(dict.fromkeys(soft_s))

hard_s=list(dict.fromkeys(hard_s))


hard_skills=hard_s
soft_skills=soft_s
feature_array = vectorizer.get_feature_names()
# number of overall model features
features_numbers = len(feature_array) #27291the most representative words/tokens
## max sorted features number
n_max = int(features_numbers * 0.1) #2729
 


output = pd.DataFrame()

for i in range(0,len(lg2.classes_)):
    print("\n****" ,lg2.classes_[i],"****\n")
    class_prob_indices_sorted = lg2.coef_[i, :].argsort()[::-1]
    raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
    print("list of unprocessed skills :")
    print(raw_skills)
    
    ## Extract hard skills
    top_hard_skills= list(set(hard_skills).intersection(raw_skills))[:20]
    print("Top technical skills",top_hard_skills) 
    
    ## Extract adjectives
    
    # Delete hard skills from raw skills list
    
    raw_skills = [x for x in raw_skills if x not in top_hard_skills]
    raw_skills = list(set(raw_skills) - set(top_hard_skills))
    
    top_soft_skills= list(set(soft_skills).intersection(raw_skills))[:20]
    #print("Top soft skills",top_soft_skills)
    #raw_skills = [x for x in raw_skills if x not in top_soft_skills]
    #raw_skills = list(set(raw_skills) - set(top_soft_skills))
    ## transform list to string
    txt = " ".join(raw_skills)
    #blob = TextBlob(raw_skills)
   
    #top 20 adjective
    #top_soft_skills1 = [w for (w, pos) in extract(txt) if pos.startswith("JJ")][:20]
    #top_adjectives = [w for (w, pos) in rake_nltk_var.extract_keywords_from_text(text) if pos.startswith("JJ")][:20]
    #print("Top 20 adjectives: ",top_adjectives)
    
    output = output.append({'senority_level':lg2.classes_[i],
                        'hard_skills':top_hard_skills,
                        'soft_skills':top_soft_skills ,
                        "not skill":raw_skills[:20]},
                       ignore_index=True)




print(output)




output.to_csv("OUTPUT.csv")







import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('wordnet')
stop = stopwords.words('english')
newStopWords = ['and',"the","to",'other',"more"]
stop.extend(newStopWords)
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


df_entry=pd.read_csv("ENTRY.csv")
df_mid=pd.read_csv("MID.csv")
df_senior=pd.read_csv("SENIOR.csv")


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
#df_entry['Description_CLEAN']  = df_entry['Description_after_keywords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
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
#df_mid['Description_CLEAN']  = df_mid['Description_after_keywords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df_mid['Description_CLEAN']  =df_mid['Description_after_keywords'].str.replace('\d+', '')
df_mid['Description_CLEAN']  = df_mid['Description_after_keywords'].str.replace('[^\w\s]',' ')

df_senior['Description_after_keywords']=df_senior['Description_after_keywords'].apply(str)
df_senior['Description_CLEAN'] = df_senior['Description_after_keywords'].apply(cleanText)

## Lower case
df_senior['Description_CLEAN'] = df_senior['Description_after_keywords'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## digits
#df_senior['Description_CLEAN']  = df_senior['Description_after_keywords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
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


###SENIOR_LEVEL
df_senior['Tokenized_values']=[" ".join(part) for part in df_senior['Tokenized'].values]

df_senior['Level']='Senior'


frames_all=[df_senior,df_entry,df_mid]
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

from nltk import tokenize
#split the text into sentences 
import spacy

my_hard_list=pd.read_excel("HARD_SKILLS_LIST.xlsx")
my_soft_list=pd.read_excel("SOFT_SKILLS_LIST.xlsx")
hard_s=my_hard_list['Skills'].tolist()

soft_s=my_soft_list["Skills"].tolist()

soft_s= list(dict.fromkeys(soft_s))

hard_s=list(dict.fromkeys(hard_s))


hard_skills=hard_s
soft_skills=soft_s

nlp = spacy.load('en_core_web_sm')
#entry
sentences_entry=[]
for text in df_entry.Description_after_keywords:
    sentences_entry.append(tokenize.sent_tokenize(text))


flat_sent = [item for sublist in sentences_entry for item in sublist]


df_sent = pd.DataFrame(flat_sent,columns=["sentences"])

df_sent["sentences"]=df_sent["sentences"].apply(lambda x: x.strip("[").strip("]"))   
df_sent["sentences"] = df_sent["sentences"].apply(lambda x: " ".join(x.lower()for x in x.split()))
 
a=[]
for row in df_sent.sentences:
    doc = nlp(row)
    for chunk in doc.noun_chunks:
        a.append(chunk.text)
    

df_n_sent1 = pd.DataFrame(a,columns=["noun_phrases"])
df_n_sent1["level"]="entry"

#mid
sentences_mid=[]
for text in df_mid.Description_after_keywords:
    sentences_mid.append(tokenize.sent_tokenize(text))


flat_sent2 = [item for sublist in sentences_mid for item in sublist]


df_sent2 = pd.DataFrame(flat_sent2,columns=["sentences"])



df_sent2["sentences"]=df_sent2["sentences"].apply(lambda x: x.strip("[").strip("]"))   
df_sent2["sentences"] = df_sent2["sentences"].apply(lambda x: " ".join(x.lower()for x in x.split()))
 

a2=[]
for row in df_sent2.sentences:
    doc = nlp(row)
    for chunk in doc.noun_chunks:
        a2.append(chunk.text)
    

df_n_sent2= pd.DataFrame(a2,columns=["noun_phrases"])
df_n_sent2["level"]="mid"


#senior
sentences_senior=[]
for text in df_senior.Description_after_keywords:
    sentences_senior.append(tokenize.sent_tokenize(text))


flat_sent2 = [item for sublist in sentences_senior for item in sublist]


df_sent3 = pd.DataFrame(flat_sent2,columns=["sentences"])



df_sent3["sentences"]=df_sent3["sentences"].apply(lambda x: x.strip("[").strip("]"))   
df_sent3["sentences"] = df_sent3["sentences"].apply(lambda x: " ".join(x.lower()for x in x.split()))
 

a3=[]
for row in df_sent3.sentences:
    doc = nlp(row)
    for chunk in doc.noun_chunks:
        a3.append(chunk.text)
    

df_n_sent3= pd.DataFrame(a3,columns=["noun_phrases"])
df_n_sent3["level"]="senior"


o=[df_n_sent1,df_n_sent2,df_n_sent3]
df_n_sent=pd.concat(o)
df_n_sent.reset_index(drop=True, inplace=True)


hard=[]
for sublist in df_n_sent.noun_phrases:  
    
    
    intersect1=[ele for ele in hard_s if(ele in sublist)]
    
    hard.append(intersect1)
     


soft=[]
for sublist in df_n_sent.noun_phrases:  
    
    intersect1=[ele for ele in soft_s if(ele in sublist)]
    
    soft.append(intersect1)

df_n_sent["Hard_matches"]=hard

df_n_sent["Soft_matches"]=soft


l=[]
for row in df_n_sent.Hard_matches:
    le=len(row)
    l.append(le)
    
df_n_sent["Len_Hard_matches"]  =l 

l1=[]
for row in df_n_sent.Soft_matches:
    le=len(row)
    l1.append(le)
    
df_n_sent["Len_Soft_matches"]  =l1



df_n_sent.loc[df_n_sent['Len_Soft_matches'] != 0, 'TYPE'] = "skill"
df_n_sent.loc[df_n_sent['Len_Hard_matches'] !=0, 'TYPE'] = "skill" #2
#df_n_sent=np.where(df_n_sent['Len_Hard_matches']>=df_n_sent['Len_Soft_matches'], 


df_n_sent.TYPE.fillna('not_skill', inplace=True)

df_n_sent=df_n_sent.drop_duplicates(subset="noun_phrases",keep="last")
df_n_sent.reset_index(drop=True, inplace=True)
                                           #'hard', 'soft')       
"""
df_n_sent.type1.fillna(0, inplace=True)
df_n_sent.type2.fillna(0, inplace=True)

df_n_sent["check"]=df_n_sent['type1']+df_n_sent['type2']

df_n_sent.loc[df_n_sent['check'] ==0 , 'type'] = "not_skill"
df_n_sent.loc[df_n_sent['check'] ==1 , 'type'] = "soft_skill"
df_n_sent.loc[df_n_sent['check'] ==2 , 'type'] = "hard_skill"
df_n_sent.loc[df_n_sent['check'] >=3 , 'type'] = "hard+soft"
"""



import matplotlib.pyplot as plt 
plt.figure()
pd.value_counts(df_n_sent['TYPE']).plot.bar(title="Type of skill distribution in df")
plt.xlabel("type")
plt.ylabel("No. of rows in df")
plt.show()



#MODELING
    
'''
Modeling
We are now going to translate this skill-extraction problem into a classification one first. And then extract the most important features from each class.

The most important features, in this case, represent the words that most likely will belong to a class ( in our case senority level)

'''






##2nd approach Tokenized text

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




print("Accuracy score DT is: ",accuracy_score(y_test, Y_test_pred_DT)) #0.696

#MODEL EVALUATION
#evaluate the predictions
print("Accuracy score  NB is: ",accuracy_score(y_test, y_predictedCLF)) # 0.659
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedCLF))
print("Classification Report NB: ")
print(classification_report(y_test, y_predictedCLF))

print("Accuracy score  LG is: ",accuracy_score(y_test, y_predictedLG2)) #  0.725#BEST
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedLG2))
print("Classification Report LG: ")
print(classification_report(y_test, y_predictedLG2))


#3rd approach noun phrases
## Converting text to features 
vectorizer = TfidfVectorizer()
#Tokenize and build vocabulary
X = vectorizer.fit_transform(df_n_sent.level) 
y = df_n_sent.TYPE

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
print("Accuracy score  NB is: ",accuracy_score(y_test, y_predictedCLF)) #
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedCLF))
print("Classification Report NB: ")
print(classification_report(y_test, y_predictedCLF))

print("Accuracy score  LG is: ",accuracy_score(y_test, y_predictedLG1)) # 
print("Classes: (to help read Confusion Matrix)\n", clf.classes_)
print("Confusion Matrix: ")

print(confusion_matrix(y_test, y_predictedLG1))
print("Classification Report LG: ")
print(classification_report(y_test, y_predictedLG1))


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
features_numbers = len(feature_array)
## max sorted features number
n_max = int(features_numbers * 0.1)

##initialize output dataframe
output = pd.DataFrame()
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
for i in range(0,len(clf.classes_)):
    
    class_prob_indices_sorted =  clf.feature_log_prob_[i, :].argsort()[::-1]
    raw_skills = np.take(feature_array, class_prob_indices_sorted[:n_max])
    #raw_skills =raw_skills.flatten()
    print("list of unprocessed skills :")
    print(raw_skills)
    
    ## Extract hard skills
    top_hard_skills=([ele for ele in raw_skills if(ele in hard_s)])[:20]
    
    print("Top technical skills",top_hard_skills)
    
    ## Extract adjectives
    
    # Delete hard skills from raw skills list
    ## At this steps, raw skills list doesnt contain the hard skills
    #raw_skills = [x for x in raw_skills if x not in top_hard_skills]
    #raw_skills = list(set(raw_skills) - set(top_hard_skills))
    
    top_soft_skills=([ele for ele in raw_skills if(ele in soft_s)])[:20]
    print("Top soft skills",top_soft_skills)
    #raw_skills = [x for x in raw_skills if x not in top_soft_skills]
   # raw_skills = list(set(raw_skills) - set(top_soft_skills))
    # transform list to string
    txt = " ".join(raw_skills)
    blob = TextBlob(txt)
    #top 20 adjective
    top_adjectives = [w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("JJ")][:20]
    top_nouns=[w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("NN")][:20]
    #top_adjectives = [w for (w, pos) in rake_nltk_var.extract_keywords_from_text(text) if pos.startswith("JJ")][:20]
    #print("Top 20 adjectives: ",top_adjectives)
    
    output = output.append({'type_skill':clf.classes_[i],
                        "not_skill":raw_skills,
                        'soft_skills': top_adjectives,
                        "hard_skills":top_hard_skills
                        },
                       ignore_index=True)




print(output.T)



output.to_csv("OUTPUT.csv")






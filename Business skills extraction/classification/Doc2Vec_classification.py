
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils

from tqdm import tqdm
import multiprocessing
import nltk
from nltk.corpus import stopwords
import pandas as pd
#tqdm.pandas(desc="progress-bar")
import numpy as np
import re

df=pd.read_csv("Jobs_by_Senority_Level.csv")
my_hard_list=pd.read_excel("HARD_SKILLS_LIST.xlsx")
my_soft_list=pd.read_excel("SOFT_SKILLS_LIST.xlsx")
hard_s=my_hard_list['Skills'].tolist()

soft_s=my_soft_list["Skills"].tolist()



def cleanText(text):
   
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


df['Description_after_keywords']=df['Description_after_keywords'].apply(str)
df['Description_after_keywords'] = df['Description_after_keywords'].apply(cleanText)


train, test = train_test_split(df, test_size=0.3, random_state=42)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Description_after_keywords']), tags=[r.Level]), axis=1) #tagging documents per level
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Description_after_keywords']), tags=[r.Level]), axis=1)

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
y_pred1 = logreg.predict(X_test1)




print('Testing accuracy from DM algorith %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score DM algorith: {}'.format(f1_score(y_test, y_pred, average='weighted')))

print('Testing accuracy DBow algorith %s' % accuracy_score(y_test1, y_pred1))
print('Testing F1 score  DBow algorith: {}'.format(f1_score(y_test1, y_pred1, average='weighted')))




hard_skills=hard_s
soft_skills=soft_s



voc=model_dbow.wv.index_to_key 

features_numbers = len(voc) #35798
## max sorted features number
n_max = int(features_numbers*0.1) 





##initialize output dataframe
output = pd.DataFrame()
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
for i in range(0,len(logreg.classes_)):
    
    class_prob_indices_sorted = logreg1.coef_[i,:].argsort()[::-1]
    raw_skills = np.take(voc, class_prob_indices_sorted[:n_max])
    raw_skills =raw_skills.flatten()
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
    #txt = " ".join(raw_skills)
    #blob = TextBlob(txt)
    #top 20 adjective
    #top_adjectives = [w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("JJ")][:20]
    #top_nouns=[w for (w, pos) in TextBlob(txt).pos_tags if pos.startswith("NN")][:20]
    #top_adjectives = [w for (w, pos) in rake_nltk_var.extract_keywords_from_text(text) if pos.startswith("JJ")][:20]
    #print("Top 20 adjectives: ",top_adjectives)
    
    output = output.append({'type_skill':logreg.classes_[i],
                        'hard_skills':top_hard_skills,
                        'soft_skills': top_soft_skills },
                       ignore_index=True)




print(output.T)

output.T.to_csv("output.csv")

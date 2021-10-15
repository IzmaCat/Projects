
import glob
import numpy as np
import pandas as pd
from nltk import FreqDist
from nltk.corpus import stopwords

from plotly.offline import plot

import plotly.express as px




jobs = []
path="/Jobs_round 2/*csv"

for infile in glob.glob(path):
    data = pd.read_csv(infile)
   
    jobs.append(data)

all_jobs = pd.concat(jobs).reset_index(drop=True)

#all_jobs.to_csv('jobs.csv')

#check  index
all_jobs.index

all_jobs.shape

#drop none values
all_jobs = all_jobs.replace(to_replace='None', value=np.nan).dropna()

all_jobs.columns
all_jobs.shape
all_jobs.head()

#check for duplicates

all_jobs.drop_duplicates(subset ="Description", keep = "first",inplace=True)
all_jobs.reset_index(drop=True, inplace=True)
all_jobs.info()

all_jobs


'''
In general I am interested in the job description text (column->Description)
Unfortunately the majority of the job posts had very different text structures and it was difficult to 
isolate the desiring parts.
Since I do not need the whole text; just the parts mentioning Qualifications+Responsibilities
manually collected keywords from job post where I can isolate the text we want

If it find a word/phrase from the keep list it keeps evrything after that word
and if it finds a word from the throw list it throws everything after that word



'''
'''
The process below keeps the parts of the text we are interested based on the keywords
'''

#these are the lists with the keywords to "crop " our job description text

keep_list=["Job Summary","Responsibilities",'Key Responsibilities','Specific duties ',"Qualifications","Candidates profile","Duties","Job Description","Minimum qualifications"," Position Summary","Job Opportunity","Position Summary","Qualifications","Basic Qualifications","Additional Required Qualifications",
            "Preferred Qualifications","Candidates must have","Key Responsibilities","Requirements","Key Qualifications","Primary Duties and Responsibilities",
            "Required Qualifications","What you''ll be in charge of","This might be good fit if you","why our culture is unique",
            "The opportunity","Candidates profile","Collaborative Leadership","Position Requirements","Primary functions and responsibilities","Duties",
                "Desired Qualifications","Job Profile","Job Description","Job responsibilities and compentencies","Minimum Requirements ",
                "Essential job functions ","education/experience requirements","summary description","Required Skills and Experience","your role","We are seeking"]




throw_list=["We are","Institutional Information","If you are best qualified","To apply","Application Instructions","Applicants must provide","For best consideration","About","Benefits",
                   " benefits summary","policy","submission","why join us","Application process","salary","overview","contact","disclaimer statement ","about the team","the team",
                       "clearence","we value","its mission","We are growing","The mission","What we offer","What We Offfer","Equal Opportunity Employer ",
                       "our commitment","located","Location","About Us","Come and join us","Disclaimer","Please note","please note"]



keyword_txt=[] #text kept from keywords by keep list
keep_keyword=[]#keywords matched from keep_list
throw_keyword=[]#keywords matched from throw list
final_text=[]
for text in all_jobs.Description:
    res_keep = [ele for ele in keep_list if(ele in text)]
    
    res_throw=[ele for ele in throw_list if (ele in text)] 
    
    for word in res_keep:
        

        first_res_keep=(res_keep[0]) #keeps the first element that matched from the keep_list
       
    keep_keyword.append(first_res_keep)
    for ele in first_res_keep:
           
            part_a=text.split(first_res_keep,1)[-1]
    

    keyword_txt.append(part_a)#list of strings where each string is the text cropped from the matching


   
    for w in res_throw:
        
        
        first_res_throw=(res_throw[0])#keeps the first element that matched from the throw_list
        print(first_res_throw)
    throw_keyword.append(first_res_throw)  
    try: #avoiding nameErrors
        for first in first_res_throw:
            part_b=part_a.split(first_res_throw,1)[0]#takes the cropped text from above and applies the matching to the throw_list first match
        final_text.append(part_b)
    except: 
        pass
    
      
    df= pd.DataFrame(final_text,columns=["Description_after_keywords"]) #new dataframe with new text
#df.drop_duplicates(subset ="Description_after_keywords", keep = "first",inplace=True)
df.reset_index(drop=True, inplace=True)



################SKILLS MATCHING###################

'''
Now I have the final text and i can  start the keyword matching!
I have 2 lists:
    i) skills manually collected skills especially for diversity&inclusion&equity
   
    First I will find the most frequent words in job posts
 Second I will match the skills of each list seperately and then I will combine the 2 list into one big list   
 and check the final result
'''
#datacleaning
from nltk.stem import WordNetLemmatizer
from textblob import Word
## lemmatization
df['Description_after_keywords']  = df['Description_after_keywords'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

## Lower case
df['Description_after_keywords'] = df['Description_after_keywords'].apply(lambda x: " ".join(x.lower()for x in x.split()))
## remove tabulation and punctuation
df['Description_after_keywords']  = df['Description_after_keywords'].str.replace('[^\w\s]',' ')
## digits
df['Description_after_keywords']  =df['Description_after_keywords'].str.replace('\d+', '')

#remove stop words
stop = stopwords.words('english')
df['Description_after_keywords']  = df['Description_after_keywords'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


hard_skills=pd.read_excel("HARD_SKILLS_LIST.xlsx")
soft_skills=pd.read_excel("SOFT_SKILLS_lIST.xlsx")

#FIRST APPROACH LOWER CASE AND MATCHING TO RAW TEXT
hardskill= hard_skills.Skills.tolist()
softskill=soft_skills.Skills.tolist()


#****hardskills**************
#matching skills to raw text description 
#lower case skills from list and from text
match_hard_raw=[]

for part in df.Description_after_keywords:
   
     part=part.lower()
     matches_mylist = [ele for ele in hardskill if(ele in part)]
     for word in matches_mylist:
        

        skill=str(word)
        
        match_hard_raw.append(skill)  
     
unique_match_hard_raw= list(dict.fromkeys( match_hard_raw))   

#let's find the most frequent matches
f_match_hard_raw=FreqDist(match_hard_raw)

print('Frequent match skills (mylist) in raw Full job Description:',f_match_hard_raw.most_common())

match_hard=pd.DataFrame(f_match_hard_raw.most_common(),columns=["Frequent_match_skills","Frequency"])  
  
match_hard["List"]="HARD_SKILL"

#match_hard.to_csv("hard_skills_match.csv")    

#*****softskills******


match_soft_raw=[]  
for part in df.Description_after_keywords:  
    part=part.lower()
    matches_coursera= [ele for ele in softskill if(ele in part)]
   
    for word in matches_coursera:
       
        match_skill=str(word)
        match_soft_raw.append(match_skill)


unique_match_soft_raw= list(dict.fromkeys( match_soft_raw)) 
#let's find the most frequent matches
f_match_soft_raw=FreqDist(match_soft_raw) 


print('Frequent match skills (coursera list) in raw Full job Description:',f_match_soft_raw.most_common())

match_soft=pd.DataFrame(f_match_soft_raw.most_common(),columns=["Frequent_match_skills","Frequency"])  
  
match_soft["List"]="SOFT_SKILL"
#match_soft.to_csv("soft_skills_match.csv")    


#combine hard+soft dataframes
frames=[match_soft, match_hard]
h_s=pd.concat(frames)
h_s.reset_index(drop=True, inplace=True)

#h_s.to_csv("Hard+soft-matches.csv")



top20= match_soft[:20]
top20 = top20.sort_values('Frequency', ascending=True)
fig = px.bar(top20, x="Frequency", y="Frequent_match_skills", title="TOP 20 Soft Skills In Job Descriptions",
             orientation='h',color='Frequency', width=1500, height=700)
fig.update_traces(marker_color=px.colors.qualitative.Alphabet)
fig.show()
plot(fig, filename="TOP 20 Soft Skills In Job Descriptions.html")



top20= match_hard[:20]
top20 = top20.sort_values('Frequency', ascending=True)
fig = px.bar(top20, x="Frequency", y="Frequent_match_skills", title="TOP 20 Hard Skills In Job Descriptions",
             orientation='h',color='Frequency', width=1500, height=700)
fig.update_traces(marker_color=px.colors.qualitative.Dark24)

fig.show()
plot(fig, filename="TOP 20 Hard Skills In Job Descriptions.html")


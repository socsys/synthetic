#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install google')


# In[ ]:


import pandas as pd
df=pd.read_csv('Query.csv')
#df=df[:15]
df.shape


# In[ ]:


import pandas as pd
import praw
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from googlesearch import search

client_id = "***"
client_secret = "****"
user_agent = "***"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    # username=username
)


def fetch_reddit_post_and_comments(url):
    try:
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=None)
        comments = [comment.body for comment in submission.comments.list()]
        return submission.selftext, comments
    except Exception as e:
        return f"Error: {e}", []





stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

max_cosine_scores=[]
max_bleu_scores = []
max_bleu2_scores=[]
max_bleu3_scores=[]
max_meteor_scores=[]
max_rouge1_scores=[]
max_rouge2_scores=[]
max_rougeL_scores=[]

#for query in df['query_column']:
for query in df['text']:
    query_tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(query) if word.lower() not in stop_words]




#search_results = search(query, num=5)
    search_results = search(query, num=10, stop=10, pause=2)

    results_dict = {}

    max_meteor=0.0
    max_meteor_text=""
    max_bleu=0.0
    max_bleu_text=""
    max_bleu2=0.0
    max_bleu2_text=""
    max_bleu3=0.0
    max_bleu3_text=""
        
        
        
    max_cosine=0.0
    max_cosine_text=""
    max_rougue1=0.0
    max_rougue1_text=""
    max_rougue2=0.0
    max_rougue2_text=""
    max_rougueL=0.0
    max_rougueL_text=""

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for i, result in enumerate(search_results, start=1):
        column_name = f"Result {i}"
        print(f"Result {i}: {result}")
        if 'www.reddit.com' in result:
            post_content, comments = fetch_reddit_post_and_comments(result)
            results_dict[column_name] = {
                'post_content': post_content,
                'comments': comments
             
            }
        #print(len(comments))
        
            query_str = ' '.join(query_tokens).split(" ")
            result_tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(post_content) if word.lower() not in stop_words]
            result_str = ' '.join(result_tokens).split(" ")
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([query, post_content])
        
            reference = [query_tokens]  # Reference is the tokenized query
            candidate = result_tokens  # Candidate is the tokenized search result text
            
            # Calculate cosine similarity between the query and search result text
            cosine_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            print(cosine_score)
            
            # Calculate BLEU score
            bleu_score = sentence_bleu([query_tokens], result_tokens)
            print(bleu_score)
            bleu_score2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5))
        
            bleu_score3 = sentence_bleu(reference, candidate, weights=(1/3, 1/3, 1/3))
         
        # Calculate METEOR score  for post
            meteor_score = single_meteor_score(result_str, query_str)
        #print(meteor_score_value)
        # Calculate ROUGE scores
        
        #scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)

# Calculate ROUGE scores
        #rouge_scores = scorer.score(query_str, result_str)
            rouge_scores = rouge.score(query, post_content)
        
            if cosine_score>max_cosine:
                max_cosine=cosine_score
                max_cosine_text=post_content
        
            if bleu_score>max_bleu:
                max_bleu=bleu_score
                max_bleu_text=post_content
            
            if bleu_score2>max_bleu2:
                max_bleu2=bleu_score2
                max_bleu2_text=post_content  
            
            if bleu_score3>max_bleu3:
                max_bleu3=bleu_score3
                max_bleu3_text=post_content    
            
            if meteor_score>max_meteor:    
                max_meteor=meteor_score
                max_meteor_text=post_content
            
                
            if rouge_scores['rouge1'].fmeasure>max_rougue1:    
                max_rougue1=rouge_scores['rouge1'].fmeasure
                max_rougue1_text=post_content
            
            if rouge_scores['rouge2'].fmeasure>max_rougue2:    
                max_rougue2=rouge_scores['rouge2'].fmeasure
                max_rougue2_text=post_content
        
            if rouge_scores['rougeL'].fmeasure>max_rougueL:    
                max_rougueL=rouge_scores['rougeL'].fmeasure
                max_rougueL_text=post_content    
                
            
        # Calculate meteor score value for each comment separately and pick the maximum one
        
            for comment in comments:
                result_tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(comment) if word.lower() not in stop_words]
                result_str = ' '.join(result_tokens).split(" ")
                tfidf_matrix = vectorizer.fit_transform([query, comment])
                similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
                bleu_score = sentence_bleu([query_tokens], result_tokens)
                rouge_scores = rouge.score(query, comment)
   
            
        # Calculate METEOR score  for comment
                meteor_score = single_meteor_score(result_str, query_str)
            
                if cosine_score>max_cosine:
                    max_cosine=cosine_score
                    max_cosine_text=comment
        
                if bleu_score>max_bleu:
                    max_bleu=bleu_score
                    max_bleu_text=comment
                
                if bleu_score2>max_bleu2:
                    max_bleu2=bleu_score2
                    max_bleu2_text=comment  
            
                if bleu_score3>max_bleu3:
                    max_bleu3=bleu_score3
                    max_bleu3_text=comment     
            
                if meteor_score>max_meteor:    
                    max_meteor=meteor_score
                    max_meteor_text=comment
            
            
                if rouge_scores['rouge1'].fmeasure>max_rougue1:    
                    max_rougue1=rouge_scores['rouge1'].fmeasure
                    max_rougue1_text=comment
            
                if rouge_scores['rouge2'].fmeasure>max_rougue2:    
                    max_rougue2=rouge_scores['rouge2'].fmeasure
                    max_rougue2_text=comment
            
                if rouge_scores['rougeL'].fmeasure>max_rougueL:    
                    max_rougueL=rouge_scores['rougeL'].fmeasure
                    max_rougueL_text=comment  
            
        #print("max Meteor score: {}".format(max_meteor))  
        #print(max_meteor_text)      
            
        else:
            print("Skipped: Not a Reddit link\n")
          
        
        
        
    max_cosine_scores.append(max_cosine)
    max_bleu_scores.append(max_bleu)
    max_bleu2_scores.append(max_bleu2)
    max_bleu3_scores.append(max_bleu3)
    max_meteor_scores.append(max_meteor)
    max_rouge1_scores.append(max_rougue1)
    max_rouge2_scores.append(max_rougue2)
    max_rougeL_scores.append(max_rougueL)
#df = pd.DataFrame(results_dict)
df['max_cosine_scores'] = max_cosine_scores
df['max_bleu_scores'] = max_bleu_scores
df['max_bleu2_scores'] = max_bleu2_scores
df['max_bleu3_scores'] = max_bleu3_scores
df['max_meteor_scores'] = max_meteor_scores
df['max_rouge1_scores'] = max_rouge1_scores
df['max_rouge2_scores'] = max_rouge2_scores
df['max_rougeL_scores'] = max_rougeL_scores


# In[ ]:


df.to_csv('link.csv)


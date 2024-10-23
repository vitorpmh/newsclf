from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np
import pandas as pd
# Importando bibliotecas
import pandas as pd
import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('rslp')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
def read_file(file):
    filepath = 'dados_texto/' + file  
    with open(filepath, 'r', encoding="latin-1") as file:
        lines = file.readlines()
    return lines
def generate_docs(lines):
    document = []
    documents = []
    for i in range(len(lines)):
        #if 'From:' in lines[i]:
        if 'document_id:' in lines[i]:
            documents.append(document)
            document = []
        document.append(lines[i])
    documents = documents[1:]
    return documents
def clean(documents):
    headers = ["Newsgroup:", "document_id:", "From:", "Subject:",'Archive-name:',
            'From:','archive-name:','Last-modified:','Version:',
            'For information about','Telephone:','Fax:',
            'Write to:','or:','ISBN','PGP SIGNED']
    cleaned_lines = []
    cleaned_documents = []
    for document in documents:
        cleaned_lines = []
        for line in document:
            # Remove headers
            for header in headers:
                if header in line:
                    line = ''
                    continue

            if ">" in line.strip():
                line = line.replace('>','')

            if "@" in line:
                continue
                
            if line.strip() == '':
                continue
            cleaned_lines.append(line.strip())
        
        #cleaned_lines = remove_pgp_signature(cleaned_lines)
        cleaned_documents.append(cleaned_lines)

    cleaned_lines = [line for line in cleaned_lines if line]
    return cleaned_documents
def join_lines(docs):
    docs_string = []
    for doc in docs:
        docs_string.append(' '.join(doc))
    return docs_string

lines = read_file('alt.atheism.txt')
docs = generate_docs(lines)
clean_docs = clean(docs)
joined_docs = join_lines(clean_docs)
import os
data = os.listdir('dados_texto')
data_class = {data[i]:i for i in range(len(data))}

df = pd.DataFrame()
for key,val in data_class.items():
    lines = read_file(key)
    docs = generate_docs(lines)
    clean_docs = clean(docs)
    joined_docs = join_lines(clean_docs)
    _df = pd.DataFrame()
    _df['texto'] = joined_docs
    _df['label'] = [val for _ in range(len(joined_docs))]
    df = pd.concat([df,_df])
df = df.reset_index(drop=True)

lang = 'english'
def remove_stopwords(text,lang,domain_stopwords):

  stop_words = nltk.corpus.stopwords.words(lang) # lang='portuguese' or lang='english'

  s = str(text).lower() # tudo para caixa baixa
  table = str.maketrans({key: None for key in string.punctuation})
  s = s.translate(table) # remove pontuacao
  tokens = word_tokenize(s) #obtem tokens
  v = [i for i in tokens if not i in stop_words and not i in domain_stopwords and not i.isdigit()] # remove stopwords
  s = ""
  for token in v:
    if len(token) >= 4:
      s += token+" "
  return s.strip()
def stemming(text,lang):

  stemmer = PorterStemmer() # stemming para ingles

  if lang=='portuguese':
    stemmer = nltk.stem.RSLPStemmer() # stemming para portuguese

  tokens = word_tokenize(text) #obtem tokens

  sentence_stem = ''
  doc_text_stems = [stemmer.stem(i) for i in tokens]
  for stem in doc_text_stems:
    sentence_stem += stem+" "

  return sentence_stem.strip()
def preprocess(df,lang,domain_stopwords=['newsgroup','altatheism','documentid']):
    df = df['texto']
    df = df.apply(lambda x: remove_stopwords(x,lang,domain_stopwords))
    df = df.apply(lambda x: stemming(x,lang))
    return df

X_train = preprocess(df,lang)

vec = TfidfVectorizer(max_features=10_000)
X_train_vec = vec.fit_transform(X_train)
ss  = StandardScaler()
X_train_vec = ss.fit_transform(np.asarray(X_train_vec.todense()))
lr = LogisticRegression(**{'solver': 'lbfgs', 'penalty': 'l2', 'C': 0.001, 'n_jobs': 12})
lr.fit(X_train_vec, df['label'])


















#####################################################
st.title("Text Classification App")

text_input = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if text_input.strip() == "":
        st.write("Please enter some text for classification.")
    else:
        text_input = stemming(remove_stopwords(text_input,lang,domain_stopwords=['newsgroup','altatheism','documentid']),lang)
        text_transformed = vec.transform([text_input])
        
        prediction = lr.predict(text_transformed)
        
        st.write(f"Predicted Class: {prediction[0]} : {list(data_class.keys())[prediction[0]]}")

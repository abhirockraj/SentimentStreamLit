import streamlit as st
import numpy as np
import requests
import json

st.title('Sentiment analysis of tweet :smile:')

def query(payload, API_URL):
    API_TOKEN = st.secrets['ApiHf']
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
def predict(newDf):
    endPoints = ['ProsusAI/finbert','finiteautomata/bertweet-base-sentiment-analysis','elozano/tweet_sentiment_eval']
    bl=[]
    for i in endPoints:
        API_URL = "https://api-inference.huggingface.co/models/"+i
        data = query({"inputs": newDf},API_URL)
        print(data)
        l=[]
        key= [data[0][0]['label'][:3].upper(),data[0][1]['label'][:3].upper(),data[0][2]['label'][:3].upper()]
        value = [data[0][0]['score'],data[0][1]['score'],data[0][2]['score']]
        l.append(value[key.index('NEG')])
        l.append(value[key.index('NEU')])
        l.append(value[key.index('POS')])
        bl.append(l)    
    finalA = np.array(bl)
    return np.argmax(np.mean(finalA,axis=0))    
text=st.text_input(max_chars=30,placeholder="Enter one sentence",label="Enter sentense to analysis ")
if (st.button('Predict sentiment ')):
    v=predict(text)
    if(v==0):
        st.write("Negetive Tweet :thumbsdown:")
    elif(v==1):
        st.write("Neutral Tweet :neutral_face:")
    else:
        st.write("Psitive Tweet :thumbsup:")
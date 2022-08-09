pip install Keras==1.1.0
import os
#setting the backend to theano
os.environ['KERAS_BACKEND'] = 'theano'	
import pandas as pd
#importing EmotionPredictor to predict the emotions according to POMS
from emotion_predictor import EmotionPredictor
# Pandas presentation options
pd.options.display.max_colwidth = 150   
pd.options.display.width = 200          
pd.options.display.max_columns = 7      
#reads the tweets into a dataframe
df=pd.read_csv('Tweets_Final.csv')
tweets=df['Tweet Text']
#instantiating model2 to predict the emotions of tweets according to POMS
model2 = EmotionPredictor(classification='poms', setting='mc', use_unison_model=True)
#calling the predict_probabilities method to calculate the probability for each emotion under POMS
probabilities2 = model2.predict_probabilities(tweets)
#saving the probability values to a csv file
probabilities2.to_csv('poms_emotion final.csv')
    
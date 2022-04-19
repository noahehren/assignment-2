import assignment1
import pandas as pd
import pickle

X, y = assignment1.load_prepare("test") #if nothing is in here, then the model will clean the training data, see code at the start
pipeline = pickle.load(open('pipe.pkl', 'rb'))
predictions = pipeline.predict(X)
X['Predicted_Sales'] = predictions
X.to_csv('video_game_sales.csv', index = False)

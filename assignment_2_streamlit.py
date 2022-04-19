import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math
import matplotlib

df = 'video_game_sales.csv'

st.header("Predicted Video Game Sales")

page_selected = st.sidebar.radio("Menu", ["Home", "Model"]) #providing radio button for user - setting home page as default, can look at model page, data ingest tool, and an about us page

if page_selected == "Home":
    df = pd.read_csv('video_game_sales.csv') #reading the csv file with the positive and negative sentiments

    st.subheader("Count of NA Sales Predictions by Genre")
    Genre = st.sidebar.selectbox('Genre', df['Genre'].drop_duplicates().sort_values())
    df_filter = df.loc[df['Genre'] == Genre, :]
    ax = pd.crosstab(df_filter.Genre, df.Predicted_Sales).plot(kind = "bar")
    st.pyplot(ax.figure)

    st.subheader("Count of NA Sales Predictions by Publisher wrto Genre")
    Publisher = st.sidebar.selectbox('Publisher', df['Publisher'].drop_duplicates().sort_values())
    df_filter2 = df.loc[df['Publisher'] == Publisher, :]
    ax = pd.crosstab(df_filter2.Genre, df.Predicted_Sales).plot(kind = "bar")
    st.pyplot(ax.figure)

    st.subheader('Count of NA Sales Predictions by User Score')
    df_score = df
    df_score['User_Score'] = df_score['User_Score'].astype('int')
    start, end = st.sidebar.select_slider("Select User Score Range",
        df_score['User_Score'].drop_duplicates().sort_values(), 
        value=(df_score['User_Score'].min(), df_score['User_Score'].max()))   
    df_filter3 = df_score.loc[(df_score['User_Score'] >= start) & (df_score['User_Score'] <= end), :]
    ax = pd.crosstab(df_filter3['User_Score'], df_score.Predicted_Sales).plot(kind = "bar")
    st.pyplot(ax.figure)

    st.subheader("Application Uses")
    st.write("My application is displaying North American sales predictions for video games based on genre, publisher wrto genre, and user score. Each of these interactive plots can be used by video game developers to determine which market they will target for their new games.")
    st.markdown("**Count of NA Sales Predictions by Genre**")
    st.write("This graph can be used by a company to analyze sales predictions by genre. For example, if a developer is considering making either and action game or an adventure game, they can simply select both genres and compare sales predictions.")
    st.markdown("**Count of NA Sales Predictions by Publisher wrto Genre**")
    st.write("This graph can be used by an individual to help them determine which video game company to approach with their idea. For example, someone with an idea can select between EA and Sony to see how well their idea would perform depending on the genre.")
    st.write("This graph can also be used by developers for competitive analysis. For example, a developer can analyze the performance of other developers in a given genre in hopes of finding a genre that they can break into.")
    st.markdown("**Count of NA Sales Predictions by User Score**")
    st.write("This graph predicts sales by user score. A developer can use this information as a target user score reference for their upcoming games. For example, a developer will be inclined to release a game free of bugs in order to ensure their user score does not tank on the release day.")

else:
    st.subheader('About the Dataset')
    st.write("The dataset I used to train my machine learning model was a video game sales dataset, which included sales information from major markets around the world and details on the video games such as name, genre, and rating. I created the model to predict North American (NA) sales only. Furthermore, I predicted sales based on four categories - very low, low, high, very high.")
    st.write("I split my dataset into a train and test dataset to ensure consistency when both training and testing the data.")
    st.write("To prepare the dataset, I removed all other sales figures, unique values (such as the name of the video game), and columns with a significant number of missing values, such as critic score.")
    st.write("Below are samples of the unclean and clean datasets.")
    df_unclean = pd.read_csv('video_game_sales_train.csv')
    st.markdown('**Unclean data**')
    st.dataframe(df_unclean.head())
    df = pd.read_csv('video_game_Sales.csv')
    st.markdown('**Clean data**')
    st.dataframe(df.head())
    st.subheader("Exploratory Analysis")
    st.write("When exploring the data, I focused on finding ways to clean the data because I was discarding the 'year' column. Without this column, there were not many plots that I could make that would increase my understanding of the data.") 
    st.write("Before removing the year column, I first looked at the effects of inflation over the years to find a relevant year to begin modeling. I ended up choosing 2006 as my cutoff because that is the release year of the Xbox360 and PS3, which are two modern consoles.")
    
    df_year = df_unclean.groupby(by = 'Year_of_Release').sum()
    df_year['Year_of_Release'] = df_year.index
    df_year = df_year[['Year_of_Release', 'NA_Sales']]

    st.markdown("**Finding a year to start modeling on**")
    fig1 = px.bar(df_year, x = 'Year_of_Release', y = 'NA_Sales')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("**Counting NA values**")
    st.dataframe(df_unclean.isna().sum())

    st.write("As mentioned above, I wanted to predict NA_sales based on four categories. I determined how to split this dataset based on boxplot quartiles.")
    st.markdown("**Boxplot of NA_Sales - FOR EXEMPLARY PURPOSES - this plot does not correlate to the values used in the original model because the data here is not cleaned**")
    fig2 = px.box(df_unclean, y = "NA_Sales")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader('Model Building and Evaluation')
    st.write("After I cleaned the dataset, I built three different pipelines to find the optimal parameters for predicting NA_Sales.")
    st.write("First, I used the KNearestNeighbor classifier. Before applying the classifier, I needed to perform a few extra steps on the training dataset. The following steps apply to all other classifier attempts mentioned below. First, I used train_test_split to split up the training data. Then I applied a StandardScalar to normalize the NA_Sales data points. Lastly, I applied a OneHotEncoder to create factors from the categorical values. The classifier parameter was a n_neighbors of 3. The accuracy was ok, but there was room for improvement.")
    st.write("I then tried a RandomForestClassifier with a max depth of 10. Again, I got a solid accuracy and confusion matrix, but I knew there was room for improvement.")
    st.write("Lastly, I tried a DecisionTreeClassifier. For this classifier, I found the optimal parameters using hyperparameter tuning via GridSearch. This classifier returned the best accuracy consistently, so I used it in the final pipeline.")
    st.write("It is important to note that the accuracy scores for all three classifiers varied widely from test to test. This variability was due to the small testing dataset that resulted from cleaning and splitting the data. However, I am confident that with a larger dataset, the final pipeline would perform best because of the hyperparameter tuning.")
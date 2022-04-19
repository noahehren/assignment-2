# import statements 
from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# Filename of the dataset to use for training and validation
train_data = "video_game_sales_train.csv"
# Filename of test dataset to apply your model and predict outcomes 
test_data = "video_game_sales_test.csv"

#For load_prepare - creating target variable
def NA_Sales_Conv(NA_Sales):
    if NA_Sales < 0.080000:
        return 'Very Low'
    elif NA_Sales >= 0.080000 and NA_Sales < 0.190000:
        return 'Low'
    elif NA_Sales >= 0.190000 and NA_Sales < 0.492500:
        return 'High'
    else:
        return 'Very High'

# Load the trainig data, clean/prepare and obtain training and target vectors, 
def load_prepare(train_or_apply = "train"): #this means that if you don't pass anything through the function, it will clean the train.csv
    if train_or_apply == "train":
        df_train = pd.read_csv(train_data)
    else:
        df_train = pd.read_csv(test_data)
    df_train = df_train[df_train.Year_of_Release >= math.floor(df_train.Year_of_Release.mean())] #filtering out anything before 2006
    df_train.drop(columns = ['Name', 'Year_of_Release', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'], inplace=True)
    df_train.dropna(inplace=True)

    count_platform = df_train['Platform'].value_counts()
    df_train = df_train.loc[df_train['Platform'].isin(count_platform.index[count_platform>9])]
    count_rating = df_train['Rating'].value_counts()
    df_train = df_train.loc[df_train['Rating'].isin(count_rating.index[count_rating>9])]
    count_genre = df_train['Genre'].value_counts()
    df_train = df_train.loc[df_train['Genre'].isin(count_genre.index[count_genre>9])]
    df_train = df_train.groupby('Developer').filter(lambda x : len(x)>9) #putting the other method in here for my own reference
    count_publisher = df_train['Publisher'].value_counts()
    df_train = df_train.loc[df_train['Publisher'].isin(count_publisher.index[count_publisher>9])]
    
    #Need to take this out/change it so it is not in the function itself
    df_train['NA_Sales'] = df_train['NA_Sales'].apply(NA_Sales_Conv)

    df_train_vlow = df_train[df_train.NA_Sales=='Very Low']
    df_train_low = df_train[df_train.NA_Sales=='Low']
    df_train_high = df_train[df_train.NA_Sales=='High']
    df_train_vhigh = df_train[df_train.NA_Sales=='Very High']
    n_minority_class = df_train_vlow.shape[0]

    df_train_high_undersampled = resample(df_train_high, replace=False, n_samples=n_minority_class, random_state=4321)
    df_train_vhigh_undersampled = resample(df_train_vhigh, replace=False, n_samples=n_minority_class, random_state=4321)
    df_train_low_undersampled = resample(df_train_low, replace=False, n_samples=n_minority_class, random_state=4321)
    df_train = pd.concat([df_train_vlow,df_train_low_undersampled, df_train_high_undersampled, df_train_vhigh_undersampled])

    X = df_train.drop(columns = ['NA_Sales'])
    y = df_train['NA_Sales']

    return X, y

# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
def build_pipeline_1(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    column_transformer = ColumnTransformer([
        ('num', num_pipeline, ['User_Count']), #applying standard scalar to selected columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating'])
    ])
    pipeline = Pipeline([
        ('ct', column_transformer),
        ('knnCLF', KNeighborsClassifier(n_neighbors=3))
    ])

    train_X_pipe = pipeline.fit(X_train, y_train) 
    scores = cross_val_score(train_X_pipe, X_train, y_train, scoring= 'accuracy', cv = 5)
    training_accuracy = np.mean(scores)
    y_predict = pipeline.predict(X_test)
    confusion_matrix_ = confusion_matrix(y_test, y_predict)

    return training_accuracy, confusion_matrix_, pipeline

# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
def build_pipeline_2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    column_transformer = ColumnTransformer([
        ('num', num_pipeline, ['User_Count']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating'])
    ])
    pipeline = Pipeline([
        ('ct', column_transformer),
        ('rfCLF', RandomForestClassifier(max_depth=10))
    ])

    train_X_pipe = pipeline.fit(X_train, y_train) 
    scores = cross_val_score(train_X_pipe, X_train, y_train, scoring= 'accuracy', cv = 5)
    training_accuracy = np.mean(scores)
    y_predict = pipeline.predict(X_test)
    confusion_matrix_ = confusion_matrix(y_test, y_predict)

    return training_accuracy, confusion_matrix_, pipeline


# This your final and improved model pipeline
# Split it into train/validate sets
# Build a pipeline to transform the training vector and fit an appropriate machine learning model
# Validate your model accuracy using the validation set
# Save your final pipeline to a file "pipeline.pkl"   
def build_pipeline_final(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])
    column_transformer = ColumnTransformer([
        ('num', num_pipeline, ['User_Count']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating'])
    ])
    pipeline = Pipeline([
        ('ct', column_transformer),
        ('dtCLF', DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)) #Found these criterion in the Hyperparameter tuning file
    ])

    train_X_pipe = pipeline.fit(X_train, y_train) 
    scores = cross_val_score(train_X_pipe, X_train, y_train, scoring= 'accuracy', cv = 5)
    training_accuracy = np.mean(scores)
    y_predict = pipeline.predict(X_test)
    confusion_matrix_ = confusion_matrix(y_test, y_predict)

    return training_accuracy, confusion_matrix_, pipeline


# Load final pipeline (pipe.pkl) and test dataset (test_data)
# Apply the pipeline to the test data and predict outcomes
def apply_pipeline():
    X, y = load_prepare("test") #if nothing is in here, then the model will clean the training data, see code at the start
    pipeline = pickle.load(open('pipe.pkl', 'rb'))
    predictions = pipeline.predict(X)
    # return predictions or outcomes
    return predictions
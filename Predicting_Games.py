import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import model_selection
import warnings
import os

team11 = "DetroitPistons.csv"
team22= 'GoldenStateWarriors.csv'
team1 = pd.read_csv(team11)
team2 = pd.read_csv(team22)
warnings.filterwarnings('ignore')


def PredictingGame(team1, team2):
    predict1 = team1[['Tm', 'Opp']]
    predict2 = team2[['Tm', 'Opp']]
    

    Team_score1 = 'Tm'
    Team_opponent1 = 'Opp'
    Team_score2 = 'Tm'
    Team_opponent2 = 'Opp'

    X = np.array(predict1.drop([Team_score1], 1))
    y = np.array(predict1[Team_score1])
    X_Train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 20)

    X1 = np.array(predict2.drop([Team_opponent2], 1))
    y1 = np.array(predict2[Team_opponent2])
    X_Train1, X_test1, y_train1, y_test1 = model_selection.train_test_split(X1, y1, test_size = 20)

    X2 = np.array(predict2.drop([Team_score2], 1))
    y2 = np.array(predict2[Team_score2])
    X_Train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(X2, y2, test_size = 20)

    X3 = np.array(predict1.drop([Team_opponent1], 1))
    y3 = np.array(predict1[Team_opponent1])
    X_Train3, X_test3, y_train3, y_test3 = model_selection.train_test_split(X3, y3, test_size = 20)
    
    X_Train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 20)
    linear = linear_model.LinearRegression()
    linear.fit(X_Train, y_train)
    
    X_Train1, X_test1, y_train1, y_test1 = model_selection.train_test_split(X1, y1, test_size = 20)
    linear1 = linear_model.LinearRegression()
    linear1.fit(X_Train1, y_train1)
    
    X_Train2, X_test2, y_train2, y_test2 = model_selection.train_test_split(X2, y2, test_size = 20)
    linear2 = linear_model.LinearRegression()
    linear2.fit(X_Train2, y_train2)

    X_Train3, X_test3, y_train3, y_test3 = model_selection.train_test_split(X3, y3, test_size = 20)
    linear3 = linear_model.LinearRegression()
    linear3.fit(X_Train3, y_train3)

    prediction = linear.predict(X_test)
    prediction1 = linear1.predict(X_test1)

    prediction2 = linear2.predict(X_test2)
    prediction3 = linear3.predict(X_test3)
    
    for x in range(len(prediction)):
        final_point = int(abs(prediction[x] + prediction1[x])/2)

        team1_name = (team11.split(".")[0])
        
        print(f"The predicted amount of points {team1_name} scored is {final_point}")
        
        final_point1 = int(abs(prediction2[x] + prediction3[x])/2)
        team2_name = (team22.split(".")[0])
        print(f"The predicted amount of points {team2_name} scored is {final_point1}")
        
        if(final_point > final_point1):
            print("Brooklyn Nets Wins\n")
        elif(final_point1 > final_point):
            print("GoldenState Warriors win\n")
        else:
            print("Tie\n")



def main():
    condition = True
    while(condition == True):
        user_option = input("Simulate game (Type Two Teams)\nQuit\n")
        user_option = user_option.upper()
        if(user_option == 'TWO TEAMS'):
            PredictingGame(team1, team2)
            
        elif(user_option == 'QUIT'):
            condition = False
        else:
            print('Invalid Option')

main()
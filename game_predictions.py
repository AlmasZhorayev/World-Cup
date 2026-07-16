import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

results = pd.read_csv('results.csv')

team1 = input("Enter the first team: ")
team2 = input("Enter the second team: ")

results = results[results['home_team'].isin([team1, team2]) | 
                  results['away_team'].isin([team1, team2])]


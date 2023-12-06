import pandas as pd
import numpy as np

df = pd.read_csv('data/ObesityDataSet.csv')

df = df.rename(columns={"family_history_with_overweight": "Family History Of Overweight",
                        "FAVC": "High Caloric Food Consumption",
                        "FCVC": "Vegetables Consumption",
                        "NCP": "Main Meals Number",
                        "CAEC": "Food Consumption Between Meals",
                        "SMOKE": "Smoke",
                        "CH20": "Daily Water Consumption",
                        "SSC": "Calories Consumption Monitoring",
                        "FAF": "Physical Activity Frequency",
                        "TUE": "Time Using Technology Devices",
                        "CALC": "Alcohol Consumption",
                        "MTRANS": "Transportation Used",
                        "NObesidad": "Nutritional Status"})

df = df.replace({'yes': 1, 'no': 0})


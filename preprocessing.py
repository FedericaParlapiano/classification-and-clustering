import pandas as pd
import numpy as np


def processing():
    df = pd.read_csv('data/ObesityDataSet.csv')

    df['FAVC'] = df['FAVC'].replace({'yes': 1, 'no': 0})
    df['SMOKE'] = df['SMOKE'].replace({'yes': 1, 'no': 0})
    df['SCC'] = df['SCC'].replace({'yes': 1, 'no': 0})

    df['CALC'] = df['CALC'].replace({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
    df['CAEC'] = df['CAEC'].replace({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})

    df = df.rename(columns={"family_history_with_overweight": "Family History Of Overweight",
                            "FAVC": "High Caloric Food Consumption",
                            "FCVC": "Vegetables Consumption",
                            "NCP": "Main Meals Number",
                            "CAEC": "Food Consumption Between Meals",
                            "SMOKE": "Smoke",
                            "CH2O": "Daily Water Consumption",
                            "SCC": "Calories Consumption Monitoring",
                            "FAF": "Physical Activity Frequency",
                            "TUE": "Time Using Technology Devices",
                            "CALC": "Alcohol Consumption",
                            "MTRANS": "Transportation Used",
                            "NObeyesdad": "Nutritional Status"})

    df.to_csv('data\obesity_dataset_clean.csv', header=True)


if __name__ == '__main__':
    processing()

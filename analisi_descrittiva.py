import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def weight_height():
    df = pd.read_csv('data/obesity_dataset_clean.csv')

    plt.figure(figsize=(12, 6))  # Imposta le dimensioni della figura

    # calcolo larghezza del bin secondo la regola di Scott
    larghezza_bin_weight = 3.5 * np.std(df['Weight']) / (len(df['Weight']) ** (1 / 3))
    larghezza_bin_height = 3.5 * np.std(df['Height']) / (len(df['Height']) ** (1 / 3))
    # calcolo numero di bin
    numero_bin_weight = int((max(df['Weight']) - min(df['Weight'])) / larghezza_bin_weight)
    numero_bin_height = int((max(df['Height']) - min(df['Height'])) / larghezza_bin_height)

    plt.subplot(1, 2, 1)
    sns.histplot(df['Weight'], bins=numero_bin_weight, color='#2D9596', kde=True, alpha=0.3)
    plt.xlabel('Peso')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione del peso')

    plt.subplot(1, 2, 2)
    sns.histplot(df['Height'], bins=numero_bin_height, color='#82A0D8', kde=True)
    plt.xlabel('Altezza')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione dell\'altezza')

    plt.show()


def BMI():
    df = pd.read_csv('data/obesity_dataset_clean.csv')

    df['BMI'] = df['Weight'] / df['Height'] ** 2

    sns.color_palette('Paired')
    sns.kdeplot(df, x=df['BMI'], hue=df['Gender'], fill=True)
    plt.ylabel('Densità')
    plt.title('Distribuzione dell\'indice di massa corporea')

    plt.show()


def pie_chart():
    df = pd.read_csv('data/obesity_dataset_clean.csv')
    stato_nutrizionale = df['Nutritional Status'].value_counts()
    labels = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 'Obesity Type I',
              'Obesity Type II', 'Obesity Type III']
    count = [stato_nutrizionale.get('Insufficient_Weight', 0), stato_nutrizionale.get('Normal_Weight', 0),
             stato_nutrizionale.get('Overweight_Level_I', 0), stato_nutrizionale.get('Overweight_Level_II', 0),
             stato_nutrizionale.get('Obesity_Type_I', 0), stato_nutrizionale.get('Obesity_Type_II', 0),
             stato_nutrizionale.get('Obesity_Type_III', 0)]
    colors = sns.color_palette('Paired')[0:10]
    plt.pie(count, colors=colors, autopct='%.0f%%',labels=None)
    plt.legend(labels, loc='best')
    plt.show()


#pie_chart()
#weight_height()
BMI()
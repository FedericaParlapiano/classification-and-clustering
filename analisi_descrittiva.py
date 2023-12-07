from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

file = 'data/obesity_dataset_clean.csv'
obesity = pd.read_csv(file)
obesity_no_index = obesity.iloc[:, 1:]


def weight_height():
    plt.figure(figsize=(12, 6))  # Imposta le dimensioni della figura

    # calcolo larghezza del bin secondo la regola di Scott
    larghezza_bin_weight = 3.5 * np.std(obesity['Weight']) / (len(obesity['Weight']) ** (1 / 3))
    larghezza_bin_height = 3.5 * np.std(obesity['Height']) / (len(obesity['Height']) ** (1 / 3))
    # calcolo numero di bin
    numero_bin_weight = int((max(obesity['Weight']) - min(obesity['Weight'])) / larghezza_bin_weight)
    numero_bin_height = int((max(obesity['Height']) - min(obesity['Height'])) / larghezza_bin_height)

    plt.subplot(1, 2, 1)
    sns.histplot(obesity['Weight'], bins=numero_bin_weight, color='#2D9596', kde=True, alpha=0.3)
    plt.xlabel('Peso')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione del peso')

    plt.subplot(1, 2, 2)
    sns.histplot(obesity['Height'], bins=numero_bin_height, color='#82A0D8', kde=True)
    plt.xlabel('Altezza')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione dell\'altezza')
    plt.savefig('grafici/Distribuzione Altezza', bbox_inches='tight')

    plt.show()


def BMI():
    obesity['BMI'] = obesity['Weight'] / obesity['Height'] ** 2

    sns.color_palette('Paired')
    sns.kdeplot(obesity, x=obesity['BMI'], hue=obesity['Gender'], fill=True)
    plt.ylabel('Densità')
    plt.title('Distribuzione dell\'indice di massa corporea')
    plt.savefig('grafici/Distribuzione BMI', bbox_inches='tight')

    plt.show()


def pie_chart():
    stato_nutrizionale = obesity['Nutritional Status'].value_counts()
    labels = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 'Obesity Type I',
              'Obesity Type II', 'Obesity Type III']
    count = [stato_nutrizionale.get('Insufficient_Weight', 0), stato_nutrizionale.get('Normal_Weight', 0),
             stato_nutrizionale.get('Overweight_Level_I', 0), stato_nutrizionale.get('Overweight_Level_II', 0),
             stato_nutrizionale.get('Obesity_Type_I', 0), stato_nutrizionale.get('Obesity_Type_II', 0),
             stato_nutrizionale.get('Obesity_Type_III', 0)]
    colors = sns.color_palette('Paired')[0:10]
    plt.pie(count, colors=colors, autopct='%.0f%%', labels=None)
    plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('grafici/Pie Chart', bbox_inches='tight')

    plt.show()


def plot_correlation_matrix(df):
    enc = LabelEncoder()
    columns_to_encode = ["Gender", "Family History Of Overweight", "Transportation Used", "Nutritional Status"]

    for col in columns_to_encode:
        df[col] = enc.fit_transform(df[col])

    corr = df.corr()
    plt.figure(figsize=(16, 12))
    sns.color_palette('Paired')
    sns.heatmap(corr, annot=True, cmap='crest')

    plt.title('Correlation Matrix')
    plt.savefig('grafici/Correlation Matrix', bbox_inches='tight')

    plt.show()


weight_height()
BMI()
pie_chart()
plot_correlation_matrix(obesity_no_index)

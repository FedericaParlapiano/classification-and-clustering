from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

file = 'data/obesity_dataset_clean.csv'
obesity = pd.read_csv(file)
obesity_no_index = obesity.iloc[:, 1:]

def hist():
    df = pd.read_csv('data/obesity_dataset_clean.csv')

    # calcolo larghezza del bin secondo la regola di Scott
    larghezza_bin_scott = 3.5 * np.std(df['Weight']) / (len(df['Weight']) ** (1 / 3))
    # calcolo numero di bin
    numero_bin_scott = int((max(df['Weight']) - min(df['Weight'])) / larghezza_bin_scott)

    plt.hist(df['Weight'], bins=numero_bin_scott, color='blue', edgecolor='black')

    plt.xlabel('Peso')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione del peso')

    # Mostra il grafico
    plt.show()

def plot_correlation_matrix(df):
    enc = LabelEncoder()
    columns_to_encode = ["Gender", "Family History Of Overweight", "Transportation Used", "Nutritional Status"]

    for col in columns_to_encode:
        df[col] = enc.fit_transform(df[col])

    corr = df.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, annot=True, cmap='crest')

    plt.title('Correlation Matrix')
    plt.savefig('Correlation Matrix', bbox_inches='tight')
    plt.show()


plot_correlation_matrix(obesity_no_index)
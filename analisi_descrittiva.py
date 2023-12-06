import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


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

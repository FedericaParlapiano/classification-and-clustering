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
    sns.histplot(obesity['Weight'], bins=numero_bin_weight, color='#ec7c26', kde=True, alpha=0.5)
    plt.xlabel('Peso')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione del peso')

    plt.subplot(1, 2, 2)
    sns.histplot(obesity['Height'], bins=numero_bin_height, color='#009090', kde=True, alpha=0.3)
    plt.xlabel('Altezza')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione dell\'altezza')
    plt.savefig('grafici/Distribuzione Altezza e Peso', bbox_inches='tight')

    plt.show()


def BMI():
    obesity['BMI'] = obesity['Weight'] / obesity['Height'] ** 2

    sns.color_palette('Paired')
    sns.kdeplot(obesity, x=obesity['BMI'], hue=obesity['Gender'], fill=True,
                palette={'Male': '#71a5d7', 'Female': '#e8638e'})
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


def violin_chart():
    plt.figure(figsize=(13, 6))  # Imposta le dimensioni della figura
    sns.violinplot(data=obesity, x=obesity['Nutritional Status'], y=obesity['Age'], hue=obesity['Gender'], split=True,
                   gap=.1, inner="point", cut=0, bw_adjust=3.0, palette={'Male': '#71a5d7', 'Female': '#e8638e'},
                   order=['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
                          'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'])
    plt.title('Distribuzione dello stato nutrizionale per età')

    plt.savefig('grafici/Grafico a violino', bbox_inches='tight')

    plt.show()


def plot_correlation_matrix(df):
    enc = LabelEncoder()
    columns_to_encode = ["Gender", "Family History Of Overweight", "Transportation Used", "Nutritional Status"]

    for col in columns_to_encode:
        df[col] = enc.fit_transform(df[col])

    corr = df.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, cmap='crest')

    plt.title('Correlation Matrix')
    plt.savefig('grafici/Correlation Matrix', bbox_inches='tight')

    plt.show()


def plot_scatterplot():
    labels = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 'Obesity Type I',
              'Obesity Type II', 'Obesity Type III']

    data_nw = obesity[obesity["Nutritional Status"] == "Normal_Weight"]
    data_ow1 = obesity[obesity["Nutritional Status"] == "Overweight_Level_I"]
    data_ow2 = obesity[obesity["Nutritional Status"] == "Overweight_Level_II"]
    data_ob1 = obesity[obesity["Nutritional Status"] == "Obesity_Type_I"]
    data_ob2 = obesity[obesity["Nutritional Status"] == "Obesity_Type_II"]
    data_ob3 = obesity[obesity["Nutritional Status"] == "Obesity_Type_III"]
    data_iw = obesity[obesity["Nutritional Status"] == "Insufficient_Weight"]

    sns.set_palette(sns.color_palette('Paired')[0:10])

    plt.title("Dispersione peso e altezza per stato nutrizionale")
    sns.scatterplot(data=data_iw, x="Weight", y="Height")
    sns.scatterplot(data=data_nw, x="Weight", y="Height")
    sns.scatterplot(data=data_ow1, x="Weight", y="Height")
    sns.scatterplot(data=data_ow2, x="Weight", y="Height")
    sns.scatterplot(data=data_ob1, x="Weight", y="Height")
    sns.scatterplot(data=data_ob2, x="Weight", y="Height")
    sns.scatterplot(data=data_ob3, x="Weight", y="Height")

    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.legend(labels)
    plt.savefig('grafici/Scatter Plot', bbox_inches='tight')

    plt.show()


weight_height()
BMI()
pie_chart()
violin_chart()
plot_scatterplot()
plot_correlation_matrix(obesity_no_index)


def strip_plot(df):
    df['Nutritional Status'] =\
        df['Nutritional Status'].replace('Insufficient_Weight', 'Insufficient Weight')\
            .replace('Normal_Weight', 'Normal Weight').replace('Overweight_Level_I', 'Overweight Level I')\
            .replace('Overweight_Level_II', 'Overweight Level II').replace('Obesity_Type_I', 'Obesity Type I')\
            .replace('Obesity_Type_II', 'Obesity Type II').replace('Obesity_Type_III', 'Obesity Type III')
    variables = df[['Physical Activity Frequency', 'Transportation Used', 'Nutritional Status']]
    colors = sns.color_palette('Paired')[0:7]
    nutritional_status = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II',
                          'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
    sns.stripplot(data=variables, x="Physical Activity Frequency", y="Transportation Used", hue_order=nutritional_status,
                  hue="Nutritional Status",
                  palette=colors)
    plt.xlabel('Physical Activity Frequency')
    plt.ylabel('Transportation Used')
    plt.title('Strip Plot by Category')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('grafici/Strip Plot', bbox_inches='tight')
    plt.show()


#weight_height()
#BMI()
#pie_chart()
#plot_correlation_matrix(obesity_no_index)
#scatter_with_regression(obesity)
strip_plot(obesity)
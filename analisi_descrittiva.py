from sklearn.preprocessing import LabelEncoder
from pywaffle import Waffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

file = 'data/obesity_dataset_clean.csv'
file_cluster = 'clustering/clustering_results.csv'
obesity = pd.read_csv(file)
obesity_clustering = pd.read_csv(file_cluster)
obesity_clustering = obesity_clustering.iloc[:, 1:]

obesity_no_index = obesity.iloc[:, 1:]
obesity_replaced = obesity.copy()
obesity_replaced['Nutritional Status'] \
    = obesity_replaced['Nutritional Status'].replace('Insufficient_Weight', 'Insufficient Weight') \
    .replace('Normal_Weight', 'Normal Weight').replace('Overweight_Level_I', 'Overweight Level I') \
    .replace('Overweight_Level_II', 'Overweight Level II').replace('Obesity_Type_I', 'Obesity Type I') \
    .replace('Obesity_Type_II', 'Obesity Type II').replace('Obesity_Type_III', 'Obesity Type III')
order = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II',
         'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
obesity_clustering['cluster'] = obesity_clustering['cluster'].replace(0, 'Cluster 0').replace(1, 'Cluster 1').replace(2, 'Cluster 2')
cluster = ['Cluster 0', 'Cluster 1', 'Cluster 2']

data_iw = obesity_replaced[obesity_replaced['Nutritional Status'] == 'Insufficient Weight']
data_nw = obesity_replaced[obesity_replaced['Nutritional Status'] == 'Normal Weight']
data_ol1 = obesity_replaced[obesity_replaced['Nutritional Status'] == 'Overweight Level I']
data_ol2 = obesity_replaced[obesity_replaced['Nutritional Status'] == 'Overweight Level II']
data_ot1 = obesity_replaced[obesity_replaced['Nutritional Status'] == 'Obesity Type I']
data_ot2 = obesity_replaced[obesity_replaced['Nutritional Status'] == 'Obesity Type II']
data_ot3 = obesity_replaced[obesity_replaced['Nutritional Status'] == 'Obesity Type III']


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
    colors = sns.color_palette('Paired')[0:8]
    plt.pie(count, colors=colors, autopct='%.0f%%', labels=None)
    plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('grafici/Pie Chart', bbox_inches='tight')

    plt.show()

def pie_chart_clustering():
    cluster = obesity['cluster'].value_counts()
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
    count = [cluster.get(0, 0), cluster.get(1, 0),cluster.get(2, 0)]
    colors = sns.color_palette('Paired')[0:8]
    plt.pie(count, colors=colors, autopct='%.0f%%', labels=None)
    plt.legend(labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('grafici/Pie Chart Clustering', bbox_inches='tight')

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

def violin_chart_clustering():
    plt.figure(figsize=(13, 6))  # Imposta le dimensioni della figura
    sns.violinplot(data=obesity, x=obesity['cluster'], y=obesity['Weight'], hue=obesity['Gender'], split=True,
                   gap=.1, inner="point", cut=0, bw_adjust=3.0, palette={'Male': '#71a5d7', 'Female': '#e8638e'},
                   order=[0, 1, 2])
    plt.title('Distribuzione dei cluster per peso')

    plt.savefig('grafici/Grafico a violino clustering', bbox_inches='tight')
    plt.show()


def family_history_with_overweight():
    color = sns.set_palette(sns.color_palette('Paired')[0:8])
    sns.histplot(data=obesity, x=obesity['Family History Of Overweight'], hue=obesity['Nutritional Status'],
                 multiple="dodge", shrink=.8,
                 hue_order=['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
                            'Obesity_Type_I',
                            'Obesity_Type_II', 'Obesity_Type_III'], palette=color)
    plt.title('Casi di sovrappeso in famiglia per stato nutrizionale')

    plt.savefig('grafici/Family history chart', bbox_inches='tight')

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

def plot_scatterplot_clustering():
    labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']

    data_c0 = obesity[obesity["cluster"] == 0]
    data_c1 = obesity[obesity["cluster"] == 1]
    data_c2 = obesity[obesity["cluster"] == 2]

    sns.set_palette(sns.color_palette('Paired')[0:10])

    plt.title("Dispersione peso e altezza per cluster")
    sns.scatterplot(data=data_c0, x="Weight", y="Height")
    sns.scatterplot(data=data_c1, x="Weight", y="Height")
    sns.scatterplot(data=data_c2, x="Weight", y="Height")

    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.legend(labels)
    plt.savefig('grafici/Scatter Plot Clustering', bbox_inches='tight')
    plt.show()


def strip_plot(df):
    variables = df[['Physical Activity Frequency', 'Transportation Used', 'Nutritional Status']]
    colors = sns.color_palette('Paired')[0:7]
    sns.stripplot(data=variables, x="Physical Activity Frequency", y="Transportation Used", hue_order=order,
                  hue="Nutritional Status",
                  palette=colors)
    plt.xlabel('Physical Activity Frequency')
    plt.ylabel('Transportation Used')
    plt.title('Strip Plot per Stato Nutrizionale')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('grafici/Strip Plot', bbox_inches='tight')
    plt.show()


def cat_plot(df):
    df['Nutritional Status'] \
        = df['Nutritional Status'].replace('Insufficient_Weight', 'Insufficient Weight') \
        .replace('Normal_Weight', 'Normal Weight').replace('Overweight_Level_I', 'Overweight Level I') \
        .replace('Overweight_Level_II', 'Overweight Level II').replace('Obesity_Type_I', 'Obesity Type I') \
        .replace('Obesity_Type_II', 'Obesity Type II').replace('Obesity_Type_III', 'Obesity Type III')
    columns = df[['Weight', 'Nutritional Status', 'Calories Consumption Monitoring']]
    sns.catplot(data=columns, kind="swarm", x="Nutritional Status", y="Weight", hue="Calories Consumption Monitoring")
    plt.xlabel('Nutritional Status')
    plt.xticks(rotation=45)
    plt.ylabel('Weight')
    plt.savefig('grafici/Car Plot.png', bbox_inches='tight', dpi=100)
    plt.show()


def joint_plot(df):
    df['BMI'] = df['Weight'] / df['Height'] ** 2
    columns = df[['Vegetables Consumption', 'BMI', 'Nutritional Status']]
    colors = sns.color_palette('Paired')[0:7]
    sns.jointplot(data=columns, y="Vegetables Consumption", x="BMI", hue="Nutritional Status", hue_order=order,
                  palette=colors)
    plt.title('Correlation Between Vegetables Consumption and BMI', loc='center', wrap=True, pad=-20)
    plt.xlabel('Weight')
    plt.ylabel('Vegetables Consumption')
    plt.savefig('grafici/Join Plot.png', bbox_inches='tight')
    plt.show()


def waffle_charts():
    values = obesity['High Caloric Food Consumption'].value_counts()
    df_calories = {
        '1': round((values.get(1) / (values.get(1) + values.get(0)) * 100)),
        '0': round((values.get(0) / (values.get(1) + values.get(0)) * 100))
    }

    values = obesity['Smoke'].value_counts()
    df_smoke = {
        '1': round((values.get(1) / (values.get(1) + values.get(0)) * 100)),
        '0': round((values.get(0) / (values.get(1) + values.get(0)) * 100))
    }

    values = obesity['Family History Of Overweight'].value_counts()
    df_family = {
        '1': round((values.get(1) / (values.get(1) + values.get(0)) * 100)),
        '0': round((values.get(0) / (values.get(1) + values.get(0)) * 100))
    }

    values = obesity['Calories Consumption Monitoring'].value_counts()
    df_calories_monitoring = {
        '1': round((values.get(1) / (values.get(1) + values.get(0)) * 100)),
        '0': round((values.get(0) / (values.get(1) + values.get(0)) * 100))
    }



    plt.figure(
        figsize=(10, 10),
        FigureClass=Waffle,
        rows=10,
        plots={
            221: {
                'values': list(df_calories.values()), # Convert actual number to a reasonable block number
                'title': {'label': 'Percentage of people consuming' '\n' 'high caloric food', 'loc': 'left',
                          'fontsize': 15},
                'legend': {
                    'labels': list(df_calories.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1)
                },
                'font_size': 20,
            },
            222: {
                'values': list(df_smoke.values()),
                'title': {'label': 'Percentage of smokers', 'loc': 'left', 'fontsize': 15},
                'legend': {
                    'labels': list(df_smoke.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1)
                },
                'font_size': 20,
    },
            223: {
                'values': list(df_family.values()),
                'title': {'label': 'Percentage of people having' '\n' 'family history of overweight', 'loc': 'left',
                          'fontsize': 15},
                'legend': {
                    'labels': list(df_family.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1)
                },
                'font_size': 20,
            },
            224: {
                'values': list(df_calories_monitoring.values()),
                'title': {'label': 'Percentage of people monitoring' '\n' 'calories consumption', 'loc': 'left',
                          'fontsize': 15},
                'legend': {
                    'labels': list(df_family.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1)
                },
                'font_size': 20,
            },
        },

        icon_legend=True,
        icons='child-reaching',
        font_size=16
    )
    plt.savefig('grafici/waffle_chart1.png', bbox_inches='tight')
    plt.show()

    values = obesity['Alcohol Consumption'].value_counts()
    df_alcohol = {
        '0': round((values.get(0) / (values.get(1) + values.get(0) + values.get(2) + values.get(3)) * 100))+1,
        '1': round((values.get(1) / (values.get(1) + values.get(0) + values.get(2) + values.get(3)) * 100)),
        '2': round((values.get(2) / (values.get(1) + values.get(0) + values.get(2) + values.get(3)) * 100)),
        '3': round((values.get(3) / (values.get(1) + values.get(0) + values.get(2) + values.get(3)) * 100))
    }

    values = obesity['Food Consumption Between Meals'].value_counts()
    df_food_bm = {
        '0': round((values.get(0) / (values.get(1) + values.get(0) + values.get(2) + values.get(3)) * 100)),
        '1': round((values.get(1) / (values.get(1) + values.get(0) + values.get(2) + values.get(3)) * 100)),
        '2': round((values.get(2) / (values.get(1) + values.get(0) + values.get(2) + values.get(3)) * 100)),
        '3': round((values.get(3) / (values.get(1) + values.get(0) + values.get(2) + values.get(3)) * 100))
    }

    # create intervals
    bins = pd.interval_range(0, 3, freq=1)
    # assign each value in df["column"] to bin and count bin occurences
    counts = pd.cut(obesity["Vegetables Consumption"], bins).value_counts()
    # create a Series, indexed by interval midpoints and convert to dictionary
    values = pd.Series(counts.values, index=bins.mid)

    df_vegetables = {
        '0-1': round((values.get(0.5) / (values.get(0.5) + values.get(1.5) + values.get(2.5)) * 100)),
        '1-2': round((values.get(1.5) / (values.get(0.5) + values.get(1.5) + values.get(2.5)) * 100)),
        '2-3': round((values.get(2.5) / (values.get(0.5) + values.get(1.5) + values.get(2.5)) * 100)),
    }


    values = obesity['Transportation Used'].value_counts()
    total = values.get('Public_Transportation') + values.get('Walking') + values.get('Motorbike') + values.get('Bike') + values.get('Automobile')
    df_transportation = {
        'Public Transportation': round(((values.get('Public_Transportation') / total) * 100))-1,
        'Walking': round(((values.get('Walking') / total) * 100)),
        'Motorbike': round(((values.get('Motorbike') / total) * 100)),
        'Bike': round(((values.get('Bike') / total) * 100))+1,
        'Automobile': round(((values.get('Automobile') / total) * 100))-1
    }

    plt.figure(
        figsize=(10, 10),
        FigureClass=Waffle,
        rows=10,
        plots={
            221: {
                'values': list(df_alcohol.values()),  # Convert actual number to a reasonable block number
                'title': {'label': 'Alcohol consumption', 'loc': 'left',
                          'fontsize': 15},
                'legend': {
                    'labels': list(df_alcohol.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1),
                    'fontsize': 10
                },
                'font_size': 20,
            },
            222: {
                'values': list(df_food_bm.values()),
                'title': {'label': 'Food consumption between meals', 'loc': 'left', 'fontsize': 15},
                'legend': {
                    'labels': list(df_food_bm.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1)
                },
                'font_size': 20,
            },
            223: {
                'values': list(df_vegetables.values()),
                'title': {'label': 'Vegetable consumption', 'loc': 'left',
                          'fontsize': 15},
                'legend': {
                    'labels': list(df_vegetables.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1)
                },
                'font_size': 20,
            },
            224: {
                'values': list(df_transportation.values()),
                'title': {'label': 'Transportation used', 'loc': 'left',
                          'fontsize': 15},
                'legend': {
                    'labels': list(df_transportation.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1)
                },
                'font_size': 20,
            },
        },
        icon_legend=True,
        font_size=16,
        icons='child-reaching'
    )

    plt.savefig('grafici/waffle_chart2.png', bbox_inches='tight')
    plt.show()

    values = obesity['Gender'].value_counts()
    total = values.get('Female') + values.get('Male')
    df_gender = {
        'Female': round(((values.get('Female') / total) * 100)),
        'Male': round(((values.get('Male') / total) * 100)),
    }

    # create intervals
    bins = pd.interval_range(0, 4, freq=1)
    # assign each value in df["column"] to bin and count bin occurences
    counts = pd.cut(obesity["Main Meals Number"], bins).value_counts()
    # create a Series, indexed by interval midpoints and convert to dictionary
    values = pd.Series(counts.values, index=bins.mid)

    df_mainmeals = {
        '0-1': round((values.get(0.5) / (values.get(0.5) + values.get(1.5) + values.get(2.5)) * 100)),
        '1-2': round((values.get(1.5) / (values.get(0.5) + values.get(1.5) + values.get(2.5)) * 100)),
        '2-3': round((values.get(2.5) / (values.get(0.5) + values.get(1.5) + values.get(2.5)) * 100)),
        '3-4': round((values.get(2.5) / (values.get(0.5) + values.get(1.5) + values.get(2.5)) * 100)),
    }

    plt.figure(
        figsize=(10, 10),
        FigureClass=Waffle,
        rows=10,
        plots={
            121: {
                'values': list(df_gender.values()),  # Convert actual number to a reasonable block number
                'title': {'label': 'Gender', 'loc': 'left',
                          'fontsize': 15},
                'legend': {
                    'labels': list(df_gender.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1),
                    'fontsize': 15
                },
                'font_size': 20,
            },
            122: {
                'values': list(df_mainmeals.values()),
                'title': {'label': 'Main meals number', 'loc': 'left', 'fontsize': 15},
                'legend': {
                    'labels': list(df_mainmeals.keys()),
                    'loc': 'upper left',
                    'bbox_to_anchor': (1, 1)
                },
                'font_size': 20,
            },
        },
        icon_legend=True,
        font_size=16,
        icons='child-reaching'
    )

    plt.savefig('grafici/waffle_chart3.png', bbox_inches='tight')
    plt.show()


def strip_plot_water(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('Paired')[0:7]
    sns.stripplot(x='Nutritional Status', y='Daily Water Consumption', data=df, palette=colors, jitter=True, alpha=0.5,
                  order=order)
    plt.title('Strip Plot of Daily Water Consumption and Nutritional Status')
    plt.savefig('grafici/water consumption.png', bbox_inches='tight')
    plt.show()


def distributed_dot_plot(df):
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    ax.hlines(y=order, xmin=0, xmax=df['Time Using Technology Devices'].max(), color='gray', alpha=0.5, linewidth=.5,
              linestyles='dashdot')

    for i, status in enumerate(order):
        df_status = df[df['Nutritional Status'] == status]
        ax.scatter(x=df_status['Time Using Technology Devices'], y=np.repeat(i, df_status.shape[0]),
                   s=75, edgecolors='gray', c='w', alpha=0.5)
        ax.scatter(x=df_status['Time Using Technology Devices'].median(), y=i, s=75, c='firebrick')

    red_patch = plt.plot([], [], marker="o", ms=10, ls="", mec=None, color='firebrick', label="Median")
    plt.legend(handles=red_patch)
    ax.set_title('Distribution of Time Using Technology by Nutritional Status', fontdict={'size': 22})
    ax.set_ylabel('Nutritional Status', alpha=0.7)
    ax.set_xticks(np.arange(0, df['Time Using Technology Devices'].max() + 2, 2))
    ax.set_xlabel('Time Using Technology Devices', alpha=0.7)
    ax.set_yticks(np.arange(len(df['Nutritional Status'].unique())))
    ax.set_yticklabels(df['Nutritional Status'].unique(), fontdict={'horizontalalignment': 'right'}, alpha=0.7)
    plt.xticks(alpha=0.7)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.grid(axis='both', alpha=.4, linewidth=.1)
    plt.savefig('grafici/distribution of time using technology', bbox_inches='tight')
    plt.show()


def histograms():
    fig, axes = plt.subplots(2, 3)
    colors = sns.color_palette('Paired')[0:7]
    sns.histplot(obesity_replaced, x='Age', hue="Nutritional Status", element="step", palette=colors, hue_order=order,
                 ax=axes[0, 0], legend=False).set(ylabel=None)
    axes[0, 0].set_title('Age Distribution')
    sns.histplot(obesity_replaced, x='Height', hue="Nutritional Status", element="step", palette=colors,
                 hue_order=order, ax=axes[0, 1], legend=False).set(ylabel=None)
    axes[0, 1].set_title('Height Distribution')
    sns.histplot(obesity_replaced, x='Weight', hue="Nutritional Status", element="step", palette=colors,
                 hue_order=order, ax=axes[0, 2], legend=False).set(ylabel=None)
    axes[0, 2].set_title('Weight Distribution')
    sns.histplot(obesity_replaced, x='Daily Water Consumption', hue="Nutritional Status", element="step",
                 palette=colors, hue_order=order, ax=axes[1, 0], legend=False).set(ylabel=None)
    axes[1, 0].set_title('Daily Water Consumption Distribution')
    sns.histplot(obesity_replaced, x='Physical Activity Frequency', hue="Nutritional Status", element="step",
                 palette=colors, hue_order=order, ax=axes[1, 1], legend=False).set(ylabel=None)
    axes[1, 1].set_title('Physical Activity Frequency Distribution')
    sns.histplot(obesity_replaced, x='Time Using Technology Devices', hue="Nutritional Status", element="step",
                 palette=colors, hue_order=order, ax=axes[1, 2], legend=False).set(ylabel=None)
    axes[1, 2].set_title('Time Using Technology Devices')
    fig.legend(labels=order, loc="upper right", fontsize=8.7)
    fig.text(0.07, 0.5, 'Count', va='center', rotation='vertical')
    plt.show()

def histograms_clustering():
    fig, axes = plt.subplots(2, 3)
    colors = sns.color_palette('Paired')[0:7]
    sns.histplot(obesity_clustering, x='Age', hue="cluster", element="step", palette=colors, hue_order=cluster,
                 ax=axes[0, 0], legend=False).set(ylabel=None)
    axes[0, 0].set_title('Age Distribution')
    sns.histplot(obesity_clustering, x='Height', hue="cluster", element="step", palette=colors, hue_order=cluster,
                 ax=axes[0, 1], legend=False).set(ylabel=None)
    axes[0, 1].set_title('Height Distribution')
    sns.histplot(obesity_clustering, x='Weight', hue="cluster", element="step", palette=colors,
                 hue_order=cluster, ax=axes[0, 2], legend=False).set(ylabel=None)
    axes[0, 2].set_title('Weight Distribution')
    sns.histplot(obesity_clustering, x='Daily Water Consumption', hue="cluster", element="step",
                 palette=colors, hue_order=cluster, ax=axes[1, 0], legend=False).set(ylabel=None)
    axes[1, 0].set_title('Daily Water Consumption Distribution')
    sns.histplot(obesity_clustering, x='Physical Activity Frequency', hue="cluster", element="step",
                 palette=colors, hue_order=cluster, ax=axes[1, 1], legend=False).set(ylabel=None)
    axes[1, 1].set_title('Physical Activity Frequency Distribution')
    sns.histplot(obesity_clustering, x='Time Using Technology Devices', hue="cluster", element="step",
                 palette=colors, hue_order=cluster, ax=axes[1, 2], legend=False).set(ylabel=None)
    axes[1, 2].set_title('Time Using Technology Devices')
    fig.legend(labels=cluster, loc="upper right", fontsize=8.7)
    fig.text(0.07, 0.5, 'Count', va='center', rotation='vertical')
    plt.show()

histograms_clustering()

# histograms()
# weight_height()
# BMI()
# violin_chart()
# pie_chart()
# plot_correlation_matrix(obesity_no_index)
# strip_plot(obesity)
# plot_scatterplot()
# cat_plot(obesity_replaced)
#joint_plot(obesity_replaced)
# waffle_chart("High Caloric Food Consumption")

# waffle_chart("High Caloric Food Consumption")

#distributed_dot_plot(obesity_replaced)
#waffle_charts()

'''pie_chart_clustering()
violin_chart_clustering()
plot_scatterplot_clustering()

'''
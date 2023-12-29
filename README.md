<h1 align="center"> Obesity or CVD risk - Classificazione e Clustering </h1>

Questa repository è relativa all'uso di alcune librerie per la creazione di modelli di classificazione e clustering.

## Il dataset

Il dataset considerato è consultabile al [link](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster/data). 
I dati in esso contenuti sono relativi ai livelli di obesità di alcuni individui che hanno partecipato, in forma anonima, ad un sondaggio su una piattaforma web; le risposte fornite sono state elaborate ottenendo un dataset con 17 feature e 2111 record. I soggetti prevengono da paesi del Messico, del Perù e della Colombia, con un’età compresa tra i 14 e i 61 anni, con abitudini alimentari e condizioni fisiche diverse.

Le librerie utilizzate per condurre le analisi sono Pandas, Matplotlib, Seaborn, Scikit-learn e Statsmodels.

Lo stato nutrizionale di un individuo può essere valutato in base al valore del Body Mass Index (BMI). Questo indice è calcolato dividendo il peso corporeo in chilogrammi di un individuo per il quadrato della sua statura in
metri. Sulla base del BMI si distinguono le seguenti categorie:
* sottopeso (BM I ≤ 18.5),
* normopeso (18.5 < BM I ≤ 24.9),
* sovrappeso (25.0 ≤ BM I ≤ 29.9),
* obesità di grado I (30.0 ≤ BM I ≤ 34.9),
* obesità di grado II (35.0 ≤ BM I ≤ 39.9),
* obesità di grado III (BM I ≥ 40).

Queste sono le etichette considerate per il task di classificazione.

Dopo una prima fase di ETL e di analisi descrittiva, si sono addestrati i modelli.

## Classificazione

Gli algoritmi di machine learning che sono stati utilizzati al fine di ottenere dei classificatori sono:
* Decision Tree,
* Random Forest,
* Gradient Boosting,
* XGBoost,
* AdaBoost,
* Support Vector Machine,
* Logistic Regression.

Per trovare la migliore combinazione di iperparametri si è utilizzata la Grid Search.

## Clustering

I modelli addestrati per il clustering sono:
* DBSCAN,
* K-means

usando diverse configurazioni di parametri.
Per determinare il miglior valore di k per il K-means e per scegliere l'eps per il DBSCAN si è usato il metodo del gomito.
Le prestazioni sono state valutate attraverso diverse metriche, tra cui la silhouette, la V-measure, l'omogeneità...

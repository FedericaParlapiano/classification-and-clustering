<h1 align="center"> Obesity or CVD risk - Classificazione e Clustering </h1>

## Il dataset

Il dataset considerato è consultabile al [link](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster/data). 
I dati in esso contenuti sono relativi ai livelli di obesità di alcune persone che hanno partecipato, in forma anonima, ad un sondaggio su una piattaforma web; le risposte fornite sono state elaborate ottenendo un dataset con 17 feature e 2111 record.

Le librerie utilizzate per condurre le analisi sono Pandas, Matplotlib, Seaborn, Scikit-learn e Statsmodels.

Lo stato nutrizionale di un individuo può essere valutato in base al valore del Body Mass Index (BMI). Questo indice è calcolato dividendo il peso corporeo in chilogrammi di un individuo per il quadrato della sua statura in
metri. Sulla base del BMI si distinguono le seguenti categorie:
* sottopeso (BM I ≤ 18.5),
* normopeso (18.5 < BM I ≤ 24.9),
* sovrappeso (25.0 ≤ BM I ≤ 29.9),
* obesità di grado I (30.0 ≤ BM I ≤ 34.9),
* obesità di grado II (35.0 ≤ BM I ≤ 39.9),
* obesità di grado III (BM I ≥ 40).

## Classification

La prima parte dell’analisi si focalizza sul task di classificazione 

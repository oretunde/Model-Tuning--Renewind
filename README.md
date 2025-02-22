# Model-Tuning-ReneWind

ReneWind” is a company working on improving the machinery/processes involved in the production of wind energy using machine learning and has collected data of generator failure of wind turbines using sensors. They have shared a ciphered version of the data, as the data collected through sensors is confidential (the type of data collected varies with companies). Data has 40 predictors, 20000 observations in the training set and 5000 in the test set. The objective is to build various classification models, tune them, and find the best one that will help identify failures so that the generators could be repaired before failing/breaking to reduce the overall maintenance cost

## Skills and Tools Covered:
* Classification Techniques: Logistic Regression, Decision Trees, Random Forest, Bagging Classifier, AdaBoost, Gradient Boosting Machine (GBM)
* Exploratory Data Analysis (EDA): Data cleaning, Outlier detection, Summary statistics, Correlation analysis
* Data Processing: Handling missing values, Feature selection, Scaling techniques (Normalization, Standardization)
* Model Building: Training and evaluation using original data, oversampled data, and undersampled data
* Pipelines: Implemented pipelines for consistent and reproducible preprocessing & model training
* Hyperparameter Tuning: RandomizedSearchCV applied to optimize models
* Model Evaluation Metrics: Precision, Recall, F1-score, Confusion Matrix
* Computational Considerations: Cost-sensitive classification, Class imbalance handling, Overfitting control
* Business Cost Analysis: Minimizing False Negatives (FN) to reduce costly generator failures
* Visualization Techniques: Feature importance plots, Confusion matrix visualization
* Tools & Libraries: Python, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Pandas, NumPy

## My Learning

This project focused on predictive maintenance for wind turbines, aiming to detect generator failures before they occur. The dataset contained 40 sensor-based features capturing machine conditions, requiring exploratory data analysis (EDA) to detect outliers and preprocess the data for modeling.

To address the imbalance in failure occurrences, models were trained using original data, oversampled data, and undersampled data to compare effectiveness. Six classification models were evaluated: Logistic Regression, Decision Trees, Random Forest, Bagging Classifier, AdaBoost, and Gradient Boosting Machine (GBM). Given the high cost of false negatives (missed failures), recall was prioritized, and the Random Forest with undersampling emerged as the best model, achieving a validation recall of 0.885 and a test data recall of 0.879 (manual approach).

To ensure scalability and consistency, a pipeline-based approach was implemented, automating data preprocessing, handling missing values, and applying transformations before predictions. The pipeline model achieved a slightly lower test recall of 0.872, but the minor 0.007 difference was acceptable given the advantages of automation and reproducibility.

Hyperparameter tuning was conducted using RandomizedSearchCV, optimizing models to improve recall while preventing overfitting. Business implications were considered by analyzing the trade-off between repair costs vs. replacement costs, ensuring the predictive maintenance model aligns with cost-efficient maintenance strategies.

Through this project, I gained deeper insights into class imbalance handling, predictive maintenance strategies, pipeline implementation for deployment, and the trade-offs in cost-sensitive classification modeling.

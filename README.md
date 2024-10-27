# Diabetes Prediction using Machine Learning Algorithms 

The project entails the factors of diabetes and health markers which help undestand whether there is a correlation of 
predicting of getting diabetes and its accuracy.

## Dataset Description

The data were collected from Iraqi society, as data were acquired from the laboratory of Medical City Hospital. Patients’ files were taken and data extracted from them and entered in to the database. The data consist of medical information, laboratory analysis. Data were also entered initially into the system. Columns are: Number of patient, Blood Sugar Level, Age, Gender, Creatinine ratio (Cr), Body Mass Index (BMI), Urea, Cholesterol (Chol), Fasting lipid profile, including total LDL, VLDL, Triglycerides (TG), HDL Cholesterol, HBA1C, and Class (Diabetic, Non-Diabetic, or Predict-Diabetic). Consists of 14 columns in total with exactly 1000 patient data.

Source: https://data.mendeley.com/datasets/wj9rwkp9c2/1

## Summary of Findings

We found that BMI, HbA1c, and Age has a positive correlation with class, meaning that they are the most informative feature for predicting diabetes with a score of 0.23, 0.26, and 0.12. Inter-feature relationships such as Urea and Creatinine, two metrics that are indicators of kidney function, have a strong correlation with each other with a score of 0.62. Cholesterol and LDL are moderately correlated. As for the dataset, the dataset was significantly imbalanced. There were 844 patients who has Diabetes, 103 for Non-Diabetic, and 53 individuals who Predict-Diabetic. Some features such as Age are fairly balanced. After oversampling, training, and tuning 4 classification models. Random Forest and SVM achieved the best performance with a near perfect 99% accuracy. This demonstrates their effectiveness in predicting diabetes. Logistic Regression and KNN also performed well. This finding shows that with a good data, features, and machine learning engineering, machine learning can be a valuable tool in predicting diabetes. This can help potential early diagnosis to support healthcare decisions. 

## Data Preprocessing

One of the first things we did was to drop unnecessary columns such as “ID” and “No_Pation” because they don’t provide any value to the analysis and modeling. Second, we standardized the column names and set all to lower case letters because column names are inconsistent (example: No_Pation, Cr, HbA1c, Cool, BMI). Next was to clean the ‘class’ column (target) because there were inconsistencies on the inputs (as mentioned in the description, data were manually inputted), expected unique values are [Y, P, N] which means DIABETIC, PREDICT-DIABETIC, and NON-DIABETIC, but we found [Y,  Y, P,  N, N], This means that there are spaces around the values. Next, a crucial part of modeling, we encoded categorical values such as Gender and Class (target), to numerical values, 0 for Female, 1 for male, 1, 2, and 3 for the three classes. After encoding, we found a missing value for the target column, stripping spaces didn’t fixed the problem for that one single value, instead we filled it with 1 (there is 1 value for the second N class, we assumed that this is the missing value). Finally, the most important part, oversampling with SMOTE. As seen in the EDA graph for the distribution of class, the dataset was significantly imbalanced (Y = 844, N = 103, P = 53). Oversampling method with SMOTE was used to make the data balanced, we decided not to under sample because the dataset is significantly few, we might lose a lot of data.  For the same reason, we didn’t remove outliers. We also standardized the data for models that are sensitive to feature scales. During our preliminary modeling, SVM achieved a lower accuracy score (around 80%), after standardizing, it significantly improved. The preprocessed data was saved to CSV format.s.

## Exploratory Data Analysis

### Visualization

![Figure 13](/assets/Figure_13.png)

According to the box plot, the majority of ages fall within the middle 50% range, which is 45 to 60 years old. A few individuals who are noticeably older or younger than the main group are shown by outliers that are over 70 and under 40.

![Figure 14](/assets/Figure_14.png)

With a few extreme outliers ranging up to 35, the central concentration of urea levels is rather modest, typically less than 10. 
This suggests that urea levels are generally low with sporadic rises.

![Figure 15](/assets/Figure_15.png)

Most creatinine values are clustered extremely close to zero, there are notable outliers that reach up to about 800, indicating that although normal levels are low, very high readings do occur occasionally.

![Figure 16](/assets/Figure_16.png)

This plot shows a fairly even distribution of HbA1c levels around the median, with a few outliers above 14. Most values fall between 6 and 10, indicating a relatively controlled distribution with occasional higher readings.

![Figure 17](/assets/Figure_17.png)

The majority of cholesterol readings fall between 4 and 6, with a few higher outliers reaching about 10. While some persons had noticeably higher cholesterol readings, the majority have moderate levels, according to the central box.


![Figure 18](/assets/Figure_18.png)

Triglyceride levels are clustered at lower values, according to the box plot, with outliers rising over 12. This suggests that most people have low triglyceride levels, while there are a few cases with extremely high levels.


![Figure 19](/assets/Figure_19.png)

HDL levels are generally very low, close to zero, with a few outliers stretching up to about 10. This suggests that most individuals have a low HDL range, with only a few showing significantly higher levels.

![Figure 20](/assets/Figure_20.png)

LDL levels appear stable, with a box plot concentration between 2 and 4, but there are outliers that go above 8, indicating occasional high LDL levels within a group primarily in the lower range.

![Figure 21](/assets/Figure_21.png)

VLDL levels are also generally low, with most values close to zero but with a significant number of outliers reaching up to about 35, suggesting rare spikes in VLDL among mostly low values.

![Figure 22](/assets/Figure_22.png)

BMI values are concentrated around 30, with a slightly wider distribution and some outliers above 40. This suggests that while most individuals are within a specific BMI range, a few have notably higher body mass indexes.

![Figure 1](/assets/Figure_1.png)

This figure shows the distribution of the different classification of patients Diabetic, Non-Diabetic, Predict-Diabetic.

![Figure 23](/assets/Figure_23.png)


Correlation Heatmap:

The correlation heatmap shows low to moderate correlations among features.
Notable correlations include:
Cr and Urea (0.62): This moderate positive correlation may suggest a physiological relationship, as both are kidney-related markers.
LDL and Chol (0.42): This positive correlation is consistent with the fact that LDL contributes to total cholesterol levels.
BMI has a moderate correlation with class (0.23), suggesting that BMI might have some predictive value in determining the diabetic class.
Overall, most correlations are low, indicating that multicollinearity may not be a major concern in this dataset.


## Model Development

We used the most common algorithms: Logistic Regression, Random Forest, Support Vector Machine (SVM), and k-Nearest Neighbors (k-NN). Implemented GridSearchCV to find the optimal parameters to achieve the best performance on training the models, set hyper parameters for tuning each model, best parameters are: {‘C’:100} for Logistic Regression, {‘max_depth’ : 20, ’n_estimators’: 100} for Random Forest, {‘C’: 100, ‘kernel’: ‘ref’} for SVM, and {’n_neighbors’: 3} for KNN. Implemented SMOTE on the data to balance the classes to prevent bias towards the majority class (Diabetic class). Load the new balanced dataset, separate features (X) and target (y). Split into training and testing sets (80% training, 20% testing).  Finally train the model using .fit().

## Model Evaluation

 Each model’s performance was
evaluated with the help of sk.learn_metrics based on
accuracy, precision, recall, and F1 score. The goal is to
reach at least 85% in each model.
Model: Logistic Regression

Accuracy: 0.96

Precision: 0.96

Recall: 0.96

F1 Score: 0.96

Classification Report:

 precision recall f1-score support
 
 1 0.97 0.95 0.96 177
 
 2 0.99 0.95 0.97 173
 
 3 0.93 0.98 0.95 157
 
 accuracy 0.96 507
 
 macro avg 0.96 0.96 0.96 507
 
weighted avg 0.96 0.96 0.96 507
============================================
======
Training and tuning Random Forest...
Model: Random Forest

Accuracy: 0.99

Precision: 0.99

Recall: 0.99

F1 Score: 0.99

Classification Report:

 precision recall f1-score support
 
 1 0.98 1.00 0.99 177
 
 2 1.00 0.98 0.99 173
 
 3 1.00 1.00 1.00 157
 
 accuracy 0.99 507
 
 macro avg 0.99 0.99 0.99 507
 
weighted avg 0.99 0.99 0.99 507

============================================
======
Training and tuning Support Vector Machine...

Model: Support Vector Machine


Accuracy: 0.99

Precision: 0.99

Recall: 0.99

F1 Score: 0.99

Classification Report:

 precision recall f1-score support
 
 1 0.98 1.00 0.99 177
 
 2 1.00 0.97 0.99 173
 
 3 0.99 1.00 1.00 157
 
 accuracy 0.99 507
 
 macro avg 0.99 0.99 0.99 507
 
weighted avg 0.99 0.99 0.99 507

============================================
======
Training and tuning k-Nearest Neighbors...

Model: k-Nearest Neighbors

Accuracy: 0.98

Precision: 0.98

Recall: 0.98

F1 Score: 0.98

Classification Report:

 precision recall f1-score support
 
 1 0.98 0.99 0.98 177
 
 2 1.00 0.94 0.97 173
 
 3 0.95 1.00 0.98 157
 
 accuracy 0.98 507
 
 macro avg 0.98 0.98 0.98 507
 
weighted avg 0.98 0.98 0.98 507
============================================
======
Each model surpassed the target.

## Conclusion

: Random Forest and SVM models
achieved a near perfect 99% accuracy, Logistic
Regression and KNN also performed well. These results
show that predicting diabetes occurrence is possible with
machine learning algorithms. Random Forest model is
recommended with its superior performance.
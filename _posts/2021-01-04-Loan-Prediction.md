---
published: true
---
<style>
   img {
       display: block;
       margin: auto;
   }
</style>

_This practice problem is from [Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/#About). I took a reference from Jason Brownlee's article: [How to Use ROC Curves and Precision-Recall Curves for Classification](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/) for plotting ROC curves, Precision-Recall Curves, calculating AUC, F1 score.<br>
The programming language is Python._<br>
---------<br>
### Predict Loan Eligibility for Dream Housing Finance Company<br>
Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.<br><br>
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers.<br>
### Dataset Information
**Train file:** CSV containing the customers for whom loan eligibility is known as 'Loan_Status'<br>
<table style="border: none;border-collapse: collapse;width:437pt;">
    <tbody>
        <tr>
            <td style="color:black;font-size:16px;font-weight:700;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;width:144pt;">Variable</td>
            <td style="color:black;font-size:16px;font-weight:700;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-left:none;width:293pt;min-width: 5px;user-select: text;">Description</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Loan_ID</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Unique Loan ID</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Gender</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Male/ Female</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Married</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Applicant married (Y/N)</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Dependents</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Number of dependents</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Education</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Applicant Education (Graduate/ Under Graduate)</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Self_Employed</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Self employed (Y/N)</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">ApplicantIncome</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Applicant income</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">CoapplicantIncome</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Coapplicant income</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">LoanAmount</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Loan amount in thousands</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Loan_Amount_Term</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Term of loan in months</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Credit_History</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">credit history meets guidelines</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Property_Area</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">Urban/ Semi Urban/ Rural</td>
        </tr>
        <tr>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;height:16.0pt;border-top:none;min-width: 5px;user-select: text;">Loan_Status</td>
            <td style="color:black;font-size:16px;font-weight:400;font-style:normal;text-decoration:none;font-family:Times, serif;text-align:general;vertical-align:bottom;border:.5pt solid windowtext;border-top:none;border-left:none;min-width: 5px;user-select: text;">(Target) Loan approved (Y/N)</td>
        </tr>
    </tbody>
</table>

### Import Modules

{% highlight python %}
import pandas as pd
import numpy as np
import seaborn as sns
import itertools

from matplotlib import pyplot as plt
import matplotlib
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

### Label Encoding
from sklearn.preprocessing import LabelEncoder

### Hyperparameter Tuning
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

### ROC curves
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

### Precision-Recall curve and F1 score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### Confusion Matrix
from sklearn.metrics import confusion_matrix

{% endhighlight %}

### Loading the Dataset

{% highlight python %}
df = pd.read_csv("Loan Prediction Dataset.csv")
df.head()
{% endhighlight %}

{% highlight python %}
df.describe()
{% endhighlight %}

{% highlight python %}
df.info()
{% endhighlight %}

### Preprocessing the Dataset

{% highlight python %}
# find the null values
df.isnull().sum()
{% endhighlight %}

{% highlight python %}
# Fill the missing values for numerical terms - mean.
# Mean vs median (better to choose median).
# Nether mean nor median is usable if the null value is more than 5%.
# If more than 5%, change the numerical into categorical.
# Imputing 
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
{% endhighlight %}

{% highlight python %}
# Fill the missing values for categorical terms - mode
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])
{% endhighlight %}

{% highlight python %}
df.isnull().sum()
{% endhighlight %}

### Exploratory Data Analysis

{% highlight python %}
# categorical attributes visualization
sns.countplot(df['Gender'])
{% endhighlight %}

[![Loan-Prediction-01-Gender.png]({{site.baseurl}}/images/Loan-Prediction-01-Gender.png)]({{site.baseurl}}/images/Loan-Prediction-01-Gender.png)<br>

{% highlight python %}
sns.countplot(df['Married'])
{% endhighlight %}

[![Loan-Prediction-02-Married.png]({{site.baseurl}}/images/Loan-Prediction-02-Married.png)]({{site.baseurl}}/images/Loan-Prediction-02-Married.png)<br>

{% highlight python %}
sns.countplot(df['Dependents'])
{% endhighlight %}

{% highlight python %}
sns.countplot(df['Education'])
{% endhighlight %}

{% highlight python %}
sns.countplot(df['Self_Employed'])
{% endhighlight %}

{% highlight python %}
sns.countplot(df['Property_Area'])
{% endhighlight %}

{% highlight python %}
sns.countplot(df['Loan_Status'])
{% endhighlight %}

{% highlight python %}
# numerical attributes visualization
sns.distplot(df["ApplicantIncome"])
{% endhighlight %}

{% highlight python %}
sns.distplot(df["CoapplicantIncome"])
{% endhighlight %}

{% highlight python %}
sns.distplot(df["LoanAmount"])
{% endhighlight %}

{% highlight python %}
sns.distplot(df['Loan_Amount_Term'])
{% endhighlight %}

{% highlight python %}
sns.distplot(df['Credit_History'])
# The value has already in a range of 0 to 1.
# Will not apply Log Transformation.
{% endhighlight %}

### Creation of New Attributes

{% highlight python %}
# Total income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()
{% endhighlight %}

### Log Transformation

{% highlight python %}
# apply log transformation to the attribute
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
sns.distplot(df["ApplicantIncomeLog"])
# ApplicationIncomeLog ranges between 5 to 12.
{% endhighlight %}

{% highlight python %}
df['LoanAmountLog'] = np.log(df['LoanAmount'])
sns.distplot(df["LoanAmountLog"])
{% endhighlight %}

{% highlight python %}
df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'])
sns.distplot(df["Loan_Amount_Term_Log"])
# Most of the values are at 6.
# But the scale is changed from 0 to 500 to 0 to 6 which is still better than before log transformation.
{% endhighlight %}

{% highlight python %}
df['Total_Income_Log'] = np.log(df['Total_Income'])
sns.distplot(df["Total_Income_Log"])
{% endhighlight %}

{% highlight python %}
df.head()
{% endhighlight %}

### Feature Scalling
After logs are applied, the ranges are similar. Hence we do not further scale data between 0 and 1.<br>
### Correlation Matrix

{% highlight python %}
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap="BuPu")
{% endhighlight %}

### Attributes for Logistic Regression

{% highlight python %}
df_all = df
{% endhighlight %}

{% highlight python %}
# drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomeLog']
# drop 'CoapplicantIncomeLog' because it has infinitive value.
df = df.drop(columns=cols, axis=1)
# axis=1 drops the column entirely.
df
{% endhighlight %}

### Attributes for Decision Tree and Random Forest

{% highlight python %}
# Select attributes which are not applied logarithm to
df_nonlog = df_all[['Gender','Married','Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area", "Loan_Status", "Total_Income"]]
df_nonlog
{% endhighlight %}

### Label Encoding

{% highlight python %}
# for Logistic Regression

cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
# initialize the LabelEncoder

for col in cols:
    df[col] = le.fit_transform(df[col])
{% endhighlight %}

{% highlight python %}
df.head()
{% endhighlight %}

{% highlight python %}
df.shape
{% endhighlight %}

{% highlight python %}
# for Decision Tree and Random Forest

cols = ['Gender',"Married", "Dependents","Education",'Self_Employed',"Property_Area","Loan_Status"]
le = LabelEncoder()
# initialize the LabelEncoder

for col in cols:
    df_nonlog[col] = le.fit_transform(df_nonlog[col])
{% endhighlight %}

{% highlight python %}
df_nonlog.head()
{% endhighlight %}

{% highlight python %}
df_nonlog.shape
{% endhighlight %}

### Preparation for Train-Test Split

{% highlight python %}
# Logistic Regression
# specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
# drop the Loan_Status from independent columns
y = df['Loan_Status']
{% endhighlight %}

{% highlight python %}
# Decision Tree and Random Forest
# specify input and output attributes
X_nonlog = df_nonlog.drop(columns=['Loan_Status'], axis=1)
# drop the Loan_Status from independent columns
y_nonlog = df_nonlog['Loan_Status']
{% endhighlight %}

### Hyperparameter Tuning (Grid Search)
I 

{% highlight python %}
def Evaluate_train_test_data(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    model.fit(x_train, y_train)
    # I want to compare the cross validation accuracy against the test data(which is only 10%) accuracy; hence I chose
    # cv=9 (9 folds)
    score = cross_val_score(model, x_train, y_train, cv=9)
    
    test_data_accuracy = model.score(x_test, y_test)*100
    cross_validation_training_data_accuracy = np.mean(score)*100

    return test_data_accuracy, cross_validation_training_data_accuracy, x_train, x_test, y_train, y_test
{% endhighlight %}

### ROC Curve and AUC

{% highlight python %}
def ROC_curve(model, x_train, x_test, y_train, y_test):
    # generate a no skill prediction (majority class)
    # My majority class here is "Loan_Status = 1" 
    ns_probs = [1 for _ in range(len(y_test))]
    
    # predict probabilities
    lr_probs = model.predict_proba(x_test)
    # keep probabilities for the positive outcome only

    lr_probs = lr_probs[:, 1]
    # calculate roc-auc scores for no-skill classifier and the trained model
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores

    return lr_auc
{% endhighlight %}

### Precision-Recall Curve and F1 Score

{% highlight python %}
def Precision_Recall_curve(model, x_train, x_test, y_train, y_test):
    # predict probabilities
    lr_probs = model.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    y_test_pred = model.predict(x_test)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    # calculate f1 score and Precision-Recall auc
    lr_f1, lr_auc = f1_score(y_test, y_test_pred), auc(lr_recall, lr_precision)
    # summarize scores

    # plot the precision-recall curves
    # calculate the [true positive rate] precision/recall for no skill model
    no_skill = len(y_test[y_test==1]) / len(y_test)  # <-- 
    
    # get precision score
    pr_score_of_precision = precision_score(y_test, y_test_pred)
    # get recall score
    pr_score_of_recall = recall_score(y_test, y_test_pred)


    return lr_f1, lr_auc, pr_score_of_precision, pr_score_of_recall
{% endhighlight %}

### Logistic Regression

{% highlight python %}
C = [100, 10, 1.0, 0.1, 0.01]
penalty = ['none', 'l1', 'l2', 'elasticnet']
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']


LR_params = [C, penalty, solver]
LR_params = list(itertools.product(*LR_params))


column_params = ["C", "penalty", "solver"]
Accuracy_params = ["Test Data Accuracy", "Cross Validation (Training) Data Accuracy"]
ROC_params = ["ROC_AUC"]
Precision_recall_params = ["PR_F1", 'PR_AUC', "PR_Precision_score", "PR_Recall_score"]
All_scores_params = Accuracy_params+ROC_params+Precision_recall_params+column_params

LR_ValueError = pd.DataFrame(columns = column_params)
LR_Accuracy = pd.DataFrame(columns = All_scores_params)

for i in range(len(LR_params)):
    try:
        
        model = LogisticRegression(C=LR_params[i][0], penalty=LR_params[i][1], solver=LR_params[i][2])
        
        test_data_accuracy, cross_validation_training_data_accuracy, x_train, x_test, y_train, y_test = Evaluate_train_test_data(model, X, y)
        
        # Call ROC_curve
        roc_auc = ROC_curve(model, x_train, x_test, y_train, y_test)

        # Call Precision_Recall_curve
        pr_f1, pr_auc, pr_score_of_precision, pr_score_of_recall = Precision_Recall_curve(model, x_train, x_test, y_train, y_test)

        lis_Accuracy = [test_data_accuracy, 
                        cross_validation_training_data_accuracy,
                        roc_auc,
                        pr_f1,
                        pr_auc, 
                        pr_score_of_precision, 
                        pr_score_of_recall,
                        LR_params[i][0],
                        LR_params[i][1],
                        LR_params[i][2]
                       ]
        
        LR_Accuracy = LR_Accuracy.append(pd.DataFrame([lis_Accuracy], columns=All_scores_params), ignore_index=True)
        
    except:
        
        ValueError_params = [LR_params[i][0],
                             LR_params[i][1],
                             LR_params[i][2]
                            ]
        
        LR_ValueError = LR_ValueError.append(pd.DataFrame([ValueError_params], columns=column_params), ignore_index=True)
{% endhighlight %}

### Decision Tree

{% highlight python %}
criterion = ['gini', 'entropy']
max_depth = list(range(1,5))
min_samples_split = list(range(2,4))
min_samples_leaf = list(range(1,5))

DT_params = [criterion, max_depth, min_samples_split, min_samples_leaf]
DT_params = list(itertools.product(*DT_params))


column_params = ["criterion", "max_depth", "min_samples_split", "min_samples_leaf"]
Accuracy_params = ["Test Data Accuracy", "Cross Validation (Training) Data Accuracy"]
ROC_params = ["ROC_AUC"]
Precision_recall_params = ["PR_F1", 'PR_AUC', "PR_Precision_score", "PR_Recall_score"]
All_scores_params = Accuracy_params+ROC_params+Precision_recall_params+column_params

DT_ValueError = pd.DataFrame(columns = column_params)
DT_Accuracy = pd.DataFrame(columns = All_scores_params)

for i in range(len(DT_params)):
    try:
        
        model = DecisionTreeClassifier(criterion=DT_params[i][0], max_depth=DT_params[i][1], min_samples_split=DT_params[i][2], 
                                       min_samples_leaf= DT_params[i][3])
        
        test_data_accuracy, cross_validation_training_data_accuracy, x_train, x_test, y_train, y_test = Evaluate_train_test_data(model, X_nonlog, y_nonlog)
        
        # Call ROC_curve
        roc_auc = ROC_curve(model, x_train, x_test, y_train, y_test)

        # Call Precision_Recall_curve
        pr_f1, pr_auc, pr_score_of_precision, pr_score_of_recall = Precision_Recall_curve(model, x_train, x_test, y_train, y_test)

        lis_Accuracy = [test_data_accuracy, 
                        cross_validation_training_data_accuracy,
                        roc_auc,
                        pr_f1,
                        pr_auc, 
                        pr_score_of_precision, 
                        pr_score_of_recall,
                        DT_params[i][0],
                        DT_params[i][1],
                        DT_params[i][2],
                        DT_params[i][3]
                       ]
        
        DT_Accuracy = DT_Accuracy.append(pd.DataFrame([lis_Accuracy], columns=All_scores_params), ignore_index=True)
        
    except:
        
        ValueError_params = [DT_params[i][0],
                        DT_params[i][1],
                        DT_params[i][2],
                        DT_params[i][3]
                            ]
        
        DT_ValueError = DT_ValueError.append(pd.DataFrame([ValueError_params], columns=column_params), ignore_index=True)
{% endhighlight %}

### Random Forest

{% highlight python %}
# n_estimators = [100]
# criterion = ['gini']
# max_features = ['auto']

n_estimators = [100, 200, 500]
criterion = ['gini', 'entropy']
max_features = ['auto','sqrt','log2']
max_depth = [5]
min_samples_split = [30, 60]    # [0.1*len(x_train)]
min_samples_leaf = [10, 20]  # do not overfit the model


RF_params = [n_estimators, criterion, max_features, max_depth, min_samples_split, min_samples_leaf]
RF_params = list(itertools.product(*RF_params))


column_params = ["n_estimators", "criterion", "max_features", "max_depth", "min_samples_split", "min_samples_leaf"]
Accuracy_params = ["Test Data Accuracy", "Cross Validation (Training) Data Accuracy"]
ROC_params = ["ROC_AUC"]
Precision_recall_params = ["PR_F1", 'PR_AUC', "PR_Precision_score", "PR_Recall_score"]
All_scores_params = Accuracy_params+ROC_params+Precision_recall_params+column_params

RF_ValueError = pd.DataFrame(columns = column_params)
RF_Accuracy = pd.DataFrame(columns = All_scores_params)

for i in range(len(RF_params)):
    try:
        
        model = RandomForestClassifier(n_estimators=RF_params[i][0], criterion=RF_params[i][1], max_features=RF_params[i][2], max_depth=RF_params[i][3], min_samples_split=RF_params[i][4], min_samples_leaf=RF_params[i][5])
        
        test_data_accuracy, cross_validation_training_data_accuracy, x_train, x_test, y_train, y_test = Evaluate_train_test_data(model, X_nonlog, y_nonlog)
        
        # Call ROC_curve
        roc_auc = ROC_curve(model, x_train, x_test, y_train, y_test)

        # Call Precision_Recall_curve
        pr_f1, pr_auc, pr_score_of_precision, pr_score_of_recall = Precision_Recall_curve(model, x_train, x_test, y_train, y_test)

        lis_Accuracy = [test_data_accuracy, 
                        cross_validation_training_data_accuracy,
                        roc_auc,
                        pr_f1,
                        pr_auc, 
                        pr_score_of_precision, 
                        pr_score_of_recall,
                        RF_params[i][0],
                        RF_params[i][1],
                        RF_params[i][2],
                        RF_params[i][3],
                        RF_params[i][4],
                        RF_params[i][5]
                       ]
        
        RF_Accuracy = RF_Accuracy.append(pd.DataFrame([lis_Accuracy], columns=All_scores_params), ignore_index=True)
        
    except:
        
        ValueError_params = [RF_params[i][0],
                             RF_params[i][1],
                             RF_params[i][2],
                             RF_params[i][3],
                             RF_params[i][4],
                             RF_params[i][5]
                            ]
        
        RF_ValueError = RF_ValueError.append(pd.DataFrame([ValueError_params], columns=column_params), ignore_index=True)
{% endhighlight %}

### Output to an Excel File

{% highlight python %}
with pd.ExcelWriter('results.xlsx') as writer:
    LR_ValueError.to_excel(writer, index = False, sheet_name='LR_ValueError')
    LR_Accuracy.to_excel(writer, index = False, sheet_name='LR_Accuracy')
    DT_ValueError.to_excel(writer, index = False, sheet_name='DT_ValueError')
    DT_Accuracy.to_excel(writer, index = False, sheet_name='DT_Accuracy')
    RF_ValueError.to_excel(writer, index = False, sheet_name='RF_ValueError')
    RF_Accuracy.to_excel(writer, index = False, sheet_name='RF_Accuracy')
{% endhighlight %}

### Confusion Matrix

{% highlight python %}
# Logistic Regression Confusion Matrix

model = LogisticRegression()
model.fit(x_train, y_train)

# Create a Confusion Matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)
{% endhighlight %}

{% highlight python %}
# Decision Tree Confusion Matrix

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Create a Confusion Matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)
{% endhighlight %}

{% highlight python %}
# Random Forest Confusion Matrix

model = RandomForestClassifier()
model.fit(x_train, y_train)

# Create a Confusion Matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)
{% endhighlight %}





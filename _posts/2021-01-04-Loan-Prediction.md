---
published: true
---
_This practice problem is from [Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/#About). The programming language is Python._<br>
---------<br>
### Predict Loan Eligibility for Dream Housing Finance company<br>
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

### Loading the Dataset

{% highlight python %}
df = pd.read_csv("Loan Prediction Dataset.csv")
df.head()
{% endhighlight %}


# Dataset Information

   Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.
   
   This is a standard supervised classification task.A classification problem where we have to predict whether a loan would be approved or not. Below is the dataset attributes with description.
   
Variable | Description
----------|--------------
Loan_ID | Unique Loan ID
Gender | Male/ Female
Married | Applicant married (Y/N)
Dependents | Number of dependents
Education | Applicant Education (Graduate/ Under Graduate)
Self_Employed | Self employed (Y/N)
ApplicantIncome | Applicant income
CoapplicantIncome | Coapplicant income
LoanAmount | Loan amount in thousands
Loan_Amount_Term | Term of loan in months
Credit_History | credit history meets guidelines
Property_Area | Urban/ Semi Urban/ Rural
Loan_Status | Loan approved (Y/N)


```python
%reset
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y


## Import Modules


```python
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

```

## Loading the Dataset


```python
df = pd.read_csv("Loan Prediction Dataset.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001002</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001003</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001005</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001006</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001008</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>614.000000</td>
      <td>614.000000</td>
      <td>592.000000</td>
      <td>600.00000</td>
      <td>564.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5403.459283</td>
      <td>1621.245798</td>
      <td>146.412162</td>
      <td>342.00000</td>
      <td>0.842199</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6109.041673</td>
      <td>2926.248369</td>
      <td>85.587325</td>
      <td>65.12041</td>
      <td>0.364878</td>
    </tr>
    <tr>
      <th>min</th>
      <td>150.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>12.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2877.500000</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3812.500000</td>
      <td>1188.500000</td>
      <td>128.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5795.000000</td>
      <td>2297.250000</td>
      <td>168.000000</td>
      <td>360.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>81000.000000</td>
      <td>41667.000000</td>
      <td>700.000000</td>
      <td>480.00000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 614 entries, 0 to 613
    Data columns (total 13 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Loan_ID            614 non-null    object 
     1   Gender             601 non-null    object 
     2   Married            611 non-null    object 
     3   Dependents         599 non-null    object 
     4   Education          614 non-null    object 
     5   Self_Employed      582 non-null    object 
     6   ApplicantIncome    614 non-null    int64  
     7   CoapplicantIncome  614 non-null    float64
     8   LoanAmount         592 non-null    float64
     9   Loan_Amount_Term   600 non-null    float64
     10  Credit_History     564 non-null    float64
     11  Property_Area      614 non-null    object 
     12  Loan_Status        614 non-null    object 
    dtypes: float64(4), int64(1), object(8)
    memory usage: 62.5+ KB


## Preprocessing the Dataset


```python
# find the null values
df.isnull().sum()
```




    Loan_ID               0
    Gender               13
    Married               3
    Dependents           15
    Education             0
    Self_Employed        32
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount           22
    Loan_Amount_Term     14
    Credit_History       50
    Property_Area         0
    Loan_Status           0
    dtype: int64




```python
# Fill the missing values for numerical terms - mean.
# Mean vs median (better to choose median).
# Nether mean nor median is usable if the null value is more than 5%.
# If more than 5%, change the numerical into categorical.
# Imputing 
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
```


```python
# Fill the missing values for categorical terms - mode
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])
```


```python
df.isnull().sum()
```




    Loan_ID              0
    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    Loan_Status          0
    dtype: int64



## Exploratory Data Analysis


```python
# categorical attributes visualization
sns.countplot(df['Gender'])
```




    <AxesSubplot:xlabel='Gender', ylabel='count'>




    
![png](output_14_1.png)
    



```python
sns.countplot(df['Married'])
```




    <AxesSubplot:xlabel='Married', ylabel='count'>




    
![png](output_15_1.png)
    



```python
sns.countplot(df['Dependents'])
```




    <AxesSubplot:xlabel='Dependents', ylabel='count'>




    
![png](output_16_1.png)
    



```python
sns.countplot(df['Education'])
```




    <AxesSubplot:xlabel='Education', ylabel='count'>




    
![png](output_17_1.png)
    



```python
sns.countplot(df['Self_Employed'])
```




    <AxesSubplot:xlabel='Self_Employed', ylabel='count'>




    
![png](output_18_1.png)
    



```python
sns.countplot(df['Property_Area'])
```




    <AxesSubplot:xlabel='Property_Area', ylabel='count'>




    
![png](output_19_1.png)
    



```python
sns.countplot(df['Loan_Status'])
```




    <AxesSubplot:xlabel='Loan_Status', ylabel='count'>




    
![png](output_20_1.png)
    



```python
# numerical attributes visualization
sns.distplot(df["ApplicantIncome"])
```




    <AxesSubplot:xlabel='ApplicantIncome', ylabel='Density'>




    
![png](output_21_1.png)
    



```python
sns.distplot(df["CoapplicantIncome"])
```




    <AxesSubplot:xlabel='CoapplicantIncome', ylabel='Density'>




    
![png](output_22_1.png)
    



```python
sns.distplot(df["LoanAmount"])
```




    <AxesSubplot:xlabel='LoanAmount', ylabel='Density'>




    
![png](output_23_1.png)
    



```python
sns.distplot(df['Loan_Amount_Term'])
```




    <AxesSubplot:xlabel='Loan_Amount_Term', ylabel='Density'>




    
![png](output_24_1.png)
    



```python
sns.distplot(df['Credit_History'])
# The value has already in a range of 0 to 1.
# Will not apply Log Transformation.
```




    <AxesSubplot:xlabel='Credit_History', ylabel='Density'>




    
![png](output_25_1.png)
    


## Creation of New Attributes


```python
# Total income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
      <th>Total_Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001002</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>146.412162</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>5849.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001003</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
      <td>6091.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001005</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>3000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001006</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>4941.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001008</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>6000.0</td>
    </tr>
  </tbody>
</table>
</div>



## Log Transformation
A preparation for Logistic Regression.<br>


```python
# apply log transformation to the attribute
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
sns.distplot(df["ApplicantIncomeLog"])
# ApplicationIncomeLog ranges between 5 to 12.
```




    <AxesSubplot:xlabel='ApplicantIncomeLog', ylabel='Density'>




    
![png](output_29_1.png)
    



```python
df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome'])
sns.distplot(df["ApplicantIncomeLog"])
```




    <AxesSubplot:xlabel='ApplicantIncomeLog', ylabel='Density'>




    
![png](output_30_1.png)
    



```python
df['LoanAmountLog'] = np.log(df['LoanAmount'])
sns.distplot(df["LoanAmountLog"])
```




    <AxesSubplot:xlabel='LoanAmountLog', ylabel='Density'>




    
![png](output_31_1.png)
    



```python
df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'])
sns.distplot(df["Loan_Amount_Term_Log"])
# Most of the values are at 6.
# But the scale is changed from 0 to 500 to 0 to 6 which is still better than before log transformation.
```




    <AxesSubplot:xlabel='Loan_Amount_Term_Log', ylabel='Density'>




    
![png](output_32_1.png)
    



```python
df['Total_Income_Log'] = np.log(df['Total_Income'])
sns.distplot(df["Total_Income_Log"])
```




    <AxesSubplot:xlabel='Total_Income_Log', ylabel='Density'>




    
![png](output_33_1.png)
    



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
      <th>Total_Income</th>
      <th>ApplicantIncomeLog</th>
      <th>CoapplicantIncomeLog</th>
      <th>LoanAmountLog</th>
      <th>Loan_Amount_Term_Log</th>
      <th>Total_Income_Log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001002</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>146.412162</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>5849.0</td>
      <td>8.674026</td>
      <td>-inf</td>
      <td>4.986426</td>
      <td>5.886104</td>
      <td>8.674026</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001003</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
      <td>6091.0</td>
      <td>8.430109</td>
      <td>7.318540</td>
      <td>4.852030</td>
      <td>5.886104</td>
      <td>8.714568</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001005</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>3000.0</td>
      <td>8.006368</td>
      <td>-inf</td>
      <td>4.189655</td>
      <td>5.886104</td>
      <td>8.006368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001006</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>4941.0</td>
      <td>7.856707</td>
      <td>7.765569</td>
      <td>4.787492</td>
      <td>5.886104</td>
      <td>8.505323</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001008</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>6000.0</td>
      <td>8.699515</td>
      <td>-inf</td>
      <td>4.948760</td>
      <td>5.886104</td>
      <td>8.699515</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Scaling
After logs are applied, the ranges are similar. Hence we do not further scale data between 0 and 1.<br>

## Correlation Matrix


```python
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap="BuPu")
```




    <AxesSubplot:>




    
![png](output_37_1.png)
    


## Attributes for Logistic Regression


```python
df_all = df
```


```python
# drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomeLog']
# drop 'CoapplicantIncomeLog' because it has infinitive value.
df = df.drop(columns=cols, axis=1)
# axis=1 drops the column entirely.
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
      <th>ApplicantIncomeLog</th>
      <th>LoanAmountLog</th>
      <th>Loan_Amount_Term_Log</th>
      <th>Total_Income_Log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>8.674026</td>
      <td>4.986426</td>
      <td>5.886104</td>
      <td>8.674026</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
      <td>8.430109</td>
      <td>4.852030</td>
      <td>5.886104</td>
      <td>8.714568</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>8.006368</td>
      <td>4.189655</td>
      <td>5.886104</td>
      <td>8.006368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>7.856707</td>
      <td>4.787492</td>
      <td>5.886104</td>
      <td>8.505323</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>8.699515</td>
      <td>4.948760</td>
      <td>5.886104</td>
      <td>8.699515</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>609</th>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
      <td>7.972466</td>
      <td>4.262680</td>
      <td>5.886104</td>
      <td>7.972466</td>
    </tr>
    <tr>
      <th>610</th>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
      <td>8.320205</td>
      <td>3.688879</td>
      <td>5.192957</td>
      <td>8.320205</td>
    </tr>
    <tr>
      <th>611</th>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>8.996157</td>
      <td>5.533389</td>
      <td>5.886104</td>
      <td>9.025456</td>
    </tr>
    <tr>
      <th>612</th>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>8.933664</td>
      <td>5.231109</td>
      <td>5.886104</td>
      <td>8.933664</td>
    </tr>
    <tr>
      <th>613</th>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
      <td>8.430109</td>
      <td>4.890349</td>
      <td>5.886104</td>
      <td>8.430109</td>
    </tr>
  </tbody>
</table>
<p>614 rows × 12 columns</p>
</div>



## Attributes for Decision Tree and Random Forest


```python
# Select attributes which are not applied logarithm to
df_nonlog = df_all[['Gender','Married','Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area", "Loan_Status", "Total_Income"]]
df_nonlog
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
      <th>Total_Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>146.412162</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>5849.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
      <td>6091.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>3000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>4941.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>6000.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>609</th>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2900</td>
      <td>0.0</td>
      <td>71.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
      <td>2900.0</td>
    </tr>
    <tr>
      <th>610</th>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4106</td>
      <td>0.0</td>
      <td>40.000000</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
      <td>4106.0</td>
    </tr>
    <tr>
      <th>611</th>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>8072</td>
      <td>240.0</td>
      <td>253.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>8312.0</td>
    </tr>
    <tr>
      <th>612</th>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>7583</td>
      <td>0.0</td>
      <td>187.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
      <td>7583.0</td>
    </tr>
    <tr>
      <th>613</th>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>4583</td>
      <td>0.0</td>
      <td>133.000000</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
      <td>4583.0</td>
    </tr>
  </tbody>
</table>
<p>614 rows × 13 columns</p>
</div>



## Label Encoding


```python
# for Logistic Regression

cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
# initialize the LabelEncoder

for col in cols:
    df[col] = le.fit_transform(df[col])

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
      <th>ApplicantIncomeLog</th>
      <th>LoanAmountLog</th>
      <th>Loan_Amount_Term_Log</th>
      <th>Total_Income_Log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>8.674026</td>
      <td>4.986426</td>
      <td>5.886104</td>
      <td>8.674026</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.430109</td>
      <td>4.852030</td>
      <td>5.886104</td>
      <td>8.714568</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>8.006368</td>
      <td>4.189655</td>
      <td>5.886104</td>
      <td>8.006368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>7.856707</td>
      <td>4.787492</td>
      <td>5.886104</td>
      <td>8.505323</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>8.699515</td>
      <td>4.948760</td>
      <td>5.886104</td>
      <td>8.699515</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (614, 12)




```python
# for Decision Tree and Random Forest

cols = ['Gender',"Married", "Dependents","Education",'Self_Employed',"Property_Area","Loan_Status"]
le = LabelEncoder()
# initialize the LabelEncoder

for col in cols:
    df_nonlog[col] = le.fit_transform(df_nonlog[col])
```


```python
df_nonlog.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
      <th>Total_Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5849</td>
      <td>0.0</td>
      <td>146.412162</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>5849.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>6091.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>3000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>4941.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.000000</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>6000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_nonlog.shape
```




    (614, 13)



## Preparation for Train-Test Split


```python
# Logistic Regression
# specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
# drop the Loan_Status from independent columns
y = df['Loan_Status']
```


```python
# Decision Tree and Random Forest
# specify input and output attributes
X_nonlog = df_nonlog.drop(columns=['Loan_Status'], axis=1)
# drop the Loan_Status from independent columns
y_nonlog = df_nonlog['Loan_Status']
```

## Hyperparameter Tuning  (Grid Search)
Jason Brownlee<br>
https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/<br>


```python
### Hyperparameter Tuning (Grid Search)

def Evaluate_train_test_data(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    model.fit(x_train, y_train)
    # I want to compare the cross validation accuracy against the test data(which is only 10%) accuracy; hence I chose
    # cv=9 (9 folds)
    score = cross_val_score(model, x_train, y_train, cv=9)
    
    test_data_accuracy = model.score(x_test, y_test)*100
    cross_validation_training_data_accuracy = np.mean(score)*100

    return test_data_accuracy, cross_validation_training_data_accuracy, x_train, x_test, y_train, y_test
```

## ROC Curve and AUC


```python
### ROC Curve and AUC

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
#     print('No Skill: ROC AUC=%.3f' % (ns_auc))
#     print('Model: ROC AUC=%.3f' % (lr_auc))

    return lr_auc
```

## Precision-Recall Curve and F1 Score


```python
### Precision-Recall Curve and F1 Score

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
#     print('Logistic: f1=%.3f PR_AUC=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    # calculate the [true positive rate] precision/recall for no skill model
    no_skill = len(y_test[y_test==1]) / len(y_test)  # <-- 
    
    # get precision score
    pr_score_of_precision = precision_score(y_test, y_test_pred)
    # get recall score
    pr_score_of_recall = recall_score(y_test, y_test_pred)
    
#     print("score_of_precision:", pr_score_of_precision)
#     print("score_of_recall:", pr_score_of_recall)

    return lr_f1, lr_auc, pr_score_of_precision, pr_score_of_recall

```

## Logistic Regression


```python
C = [100, 10, 1.0, 0.1, 0.01]
penalty = ['elasticnet']
solver = ['saga']

LR_params = [C, penalty, solver]
LR_params = list(itertools.product(*LR_params))


column_params = ["C", "penalty", "solver"]
Accuracy_params = ["Test Data Accuracy", "Cross Validation (Training) Data Accuracy"]
ROC_params = ["ROC_AUC"]
Precision_recall_params = ["PR_F1", 'PR_AUC', "PR_Precision_score", "PR_Recall_score"]
All_scores_params = Accuracy_params+ROC_params+Precision_recall_params+column_params

LR_ValueError = pd.DataFrame(columns = column_params)
LR_Accuracy = pd.DataFrame(columns = All_scores_params)

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
        

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-44-735ce7c081e1> in <module>
         16 LR_Accuracy = pd.DataFrame(columns = All_scores_params)
         17 
    ---> 18 model = LogisticRegression(C=LR_params[i][0], penalty=LR_params[i][1], solver=LR_params[i][2])
         19 
         20 test_data_accuracy, cross_validation_training_data_accuracy, x_train, x_test, y_train, y_test = Evaluate_train_test_data(model, X, y)


    NameError: name 'i' is not defined



```python
### Logistic Regression

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
 

```


```python
# # print(LR_ValueError)
# LR_ValueError.to_csv('LR_ValueError.csv', mode='w')
```


```python
# print(LR_Accuracy)
# LR_Accuracy.to_csv('LR_Accuracy.csv', mode='w')
```

## Decision Tree


```python
### Decision Tree

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
 

```


```python
# print(DT_ValueError)
# DT_ValueError.to_csv('DT_ValueError.csv', mode='w')
```


```python
# print(DT_Accuracy)
# DT_Accuracy.to_csv('DT_Accuracy.csv', mode='w')
```


```python
1=2
```

## Random Forest


```python
### RandomForest

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
 

```


```python
# print(RF_ValueError)
# RF_ValueError.to_csv('RF_ValueError.csv', mode='w')
```


```python
# print(RF_Accuracy)
# RF_Accuracy.to_csv('RF_Accuracy.csv', mode='w')
```

## Output as an Excel File


```python
with pd.ExcelWriter('results.xlsx') as writer:
    LR_ValueError.to_excel(writer, index = False, sheet_name='LR_ValueError')
    LR_Accuracy.to_excel(writer, index = False, sheet_name='LR_Accuracy')
    DT_ValueError.to_excel(writer, index = False, sheet_name='DT_ValueError')
    DT_Accuracy.to_excel(writer, index = False, sheet_name='DT_Accuracy')
    RF_ValueError.to_excel(writer, index = False, sheet_name='RF_ValueError')
    RF_Accuracy.to_excel(writer, index = False, sheet_name='RF_Accuracy')
```

## Confusion Matrix
https://www.ritchieng.com/machine-learning-evaluate-classification-model/<br>
A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.


```python
# Logistic Regression Confusion Matrix

model = LogisticRegression()
model.fit(x_train, y_train)

# Create a Confusion Matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)
```


```python
# Decision Tree Confusion Matrix

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Create a Confusion Matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)
```


```python
# Random Forest Confusion Matrix

model = RandomForestClassifier()
model.fit(x_train, y_train)

# Create a Confusion Matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True)
```


```python

```

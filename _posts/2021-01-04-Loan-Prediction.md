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

| Variable | Description |
| ----------: | --------------: |
| LoanID | Unique Loan ID |
| Gender | Male/ Female |
| Married | Applicant married (Y/N) |
| Dependents | Number of dependents |
| Education | Applicant Education (Graduate/ Under Graduate) |
| SelfEmployed | Self employed (Y/N) |
| ApplicantIncome | Applicant income |
| CoapplicantIncome | Coapplicant income |
| LoanAmount | Loan amount in thousands |
| LoanAmountTerm | Term of loan in months |
| CreditHistory | credit history meets guidelines |
| PropertyArea | Urban/ Semi Urban/ Rural |
| LoanStatus | Loan approved (Y/N) |
{:.table-striped}

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

### Code<br>
{% highlight python %}
    [code]
{% endhighlight %}



# 72522025_Churning_Customers

<!--toc:start-->

- [Overview of Problem](#overview-of-problem)
- [Milestones: ML Life Cycle](#milestones-ml-life-cycle)
- [Directory Structure](#directory-structure)
- [Chosen Features](#chosen-features)
- [Model Architecture](#model-architecture)
- [Deployment](#deployment)
- [Demo Video](#demo-video)
<!--toc:end-->

## Overview of Problem

Customer churn is a major problem and one of the most important concerns for
large companies. Due to the direct effect on the revenues of the companies,
especially in the telecom field, companies are seeking to develop means to
predict potential customer churn. Therefore, finding factors that increase
customer churn is important to take necessary actions to reduce this churn.

**GOAL:** Develop a churn prediction model that assists telecom operators in
predicting customers who are most likely subject to churn.

### Context

| Columns          | Meaning                                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| customerID       | The ID of the customer                                                                                             |
| gender           | Whether male or female                                                                                             |
| SeniorCitizen    | Whether the customer is a senior citizen or not (1, 0)                                                             |
| Partner          | Whether the customer has a partner or not (Yes, No)                                                                |
| Dependents       | Whether the customer has dependents or not (Yes, No)                                                               |
| tenure           | Number of months the customer has stayed with the company                                                          |
| PhoneService     | Whether the customer has a phone service or not (Yes, No)                                                          |
| MultipleLines    | Whether the customer has multiple lines or not (Yes, No, No phone service)                                         |
| InternetService  | Customer’s internet service provider (DSL, Fiber optic, No)                                                        |
| OnlineSecurity   | Whether the customer has online security or not (Yes, No, No internet service)                                     |
| OnlineBackup     | Whether the customer has online backup or not (Yes, No, No internet service)                                       |
| DeviceProtection | Whether the customer has device protection or not (Yes, No, No internet service)                                   |
| TechSupport      | Whether the customer has tech support or not (Yes, No, No internet service)                                        |
| StreamingTV      | Whether the customer has streaming TV or not (Yes, No, No internet service)                                        |
| StreamingMovies  | Whether the customer has streaming movies or not (Yes, No, No internet service)                                    |
| Contract         | The contract term of the customer (Month-to-month, One year, Two year)                                             |
| PaperlessBilling | Whether the customer has paperless billing or not (Yes, No)                                                        |
| PaymentMethod    | The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) |
| MonthlyCharges   | The amount charged to the customer monthly                                                                         |
| TotalCharges     | The total amount charged to the customer                                                                           |
| Churn            | Whether the customer churned or not (Yes or No)                                                                    |

## Milestones: ML Life Cycle

- [x] Data preprocessing
- [x] Feature Extraction
- [x] Exploratory Data Analysis
- [ ] Model training (MLP using functional API)
- [ ] Model Evaluation
- [ ] Deployment

## Directory Structure

- **app:** Source code for model deployment.
- **data:** Dataset
- **demo**: Demo video.
- **models:** Saved models.
- **src:** Source codes for model training. (`.py` and `.ipynb` files)

## Chosen Features

1. TotalCharges
2. MonthlyCharges
3. tenure
4. Contract
5. InternetService
6. PaymentMethod

## Model Architecture

## Deployment

- Website link:

## Demo Video

- YouTube link:

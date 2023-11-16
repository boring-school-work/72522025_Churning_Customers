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
- [x] Model training (MLP using functional API)
- [x] Model Evaluation
- [x] Deployment

## Directory Structure

- **app:** Source code for model deployment.
- **data:** Dataset
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

**N.B.** Grid search and cross validation were used to find the best hyperparameters.

### Initial Model

Creating a deep neural network.

```python
input_layer = Input(shape=(X_train.shape[1],))
dense_layer_1 = Dense(128, activation='relu')(input_layer)
dense_layer_2 = Dense(64, activation='tanh')(dense_layer_1)
dense_layer_3 = Dense(32, activation='elu')(dense_layer_2)
dense_layer_4 = Dense(16, activation='relu')(dense_layer_3)
dense_layer_5 = Dense(8, activation='tanh')(dense_layer_4)
output_layer = Dense(1, activation='sigmoid')(dense_layer_5)

# define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

# compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

### Optimized Model

One of the ways to solve the problem of overfitting is to use dropout layers. Dropout layers randomly drop some neurons during training.

```python
input_layer = Input(shape=(X_train.shape[1],))
dense_layer_1 = Dense(128, activation='relu')(input_layer)
dropout_layer_1 = Dropout(0.3)(dense_layer_1)
dense_layer_2 = Dense(64, activation='tanh')(dropout_layer_1)
dropout_layer_2 = Dropout(0.3)(dense_layer_2)
dense_layer_3 = Dense(32, activation='elu')(dropout_layer_2)
dense_layer_4 = Dense(16, activation='relu')(dense_layer_3)
dropout_layer_3 = Dropout(0.3)(dense_layer_4)
dense_layer_5 = Dense(8, activation='tanh')(dropout_layer_3)
output_layer = Dense(1, activation='sigmoid')(dense_layer_5)

# define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

# compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

### Final Model

Another way to reduce overfitting is reducing the number of neurons and hidden layers.

```python
input_layer = Input(shape=(X_train.shape[1],))
dense_layer_1 = Dense(32, activation='relu')(input_layer)
dense_layer_2 = Dense(16, activation='relu')(dense_layer_1)
dense_layer_3 = Dense(8, activation='relu')(dense_layer_2)
output_layer = Dense(1, activation='sigmoid')(dense_layer_3)

# define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)

# compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

## Deployment

- Website link: [https://72522025-churningcustomers.streamlit.app](https://72522025-churningcustomers.streamlit.app/)

## Demo Video

- YouTube link: [https://youtu.be/zpMz7l91k_4](https://youtu.be/zpMz7l91k_4)

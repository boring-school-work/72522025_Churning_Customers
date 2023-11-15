# %% [markdown]
# ## Import Relevant Libraries

# %%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from scikeras.wrappers import KerasClassifier

# %%

# %% [markdown]
# ## Load The Data

# %%
data = pd.read_csv("../data/CustomerChurn_dataset.csv")

# %% [markdown]
# ## Data Preprocessing

# %%
data.info()

# %%
# check for missing values
data.isnull().sum()

# %%
# loop through the categorical columns and replace spaces with np.nan
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].replace(' ', np.nan)

# %%
data.isnull().sum()

# %%
# convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])

# %%
# impute missing values in TotalCharges column
imputer = SimpleImputer(strategy='mean')
data["TotalCharges"] = imputer.fit_transform(data[["TotalCharges"]])

# %%
data.isnull().sum()

# %%
data.info()

# %%
# check all the uqniue values in each column
for col in data.columns:
    print(col, ":", len(data[col].unique()), "\n", data[col].unique(), "\n")

# %% [markdown]
# Data cleaning

# %%
# customerID does not provide any information about customer churn
data.drop("customerID", axis=1, inplace=True)

# %%
# having no interner/phone service is same as no
# replace No internet service and No phone service with No
for col in data.columns:
    if "No internet service" in data[col].unique():
        data[col] = data[col].replace("No internet service", "No")
    elif "No phone service" in data[col].unique():
        data[col] = data[col].replace("No phone service", "No")

# %%
# check all the uqniue values in each column
for col in data.columns:
    print(col, ":", len(data[col].unique()), "\n", data[col].unique(), "\n")

# %%
# list numerical and categorical columns
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [col for col in data.columns if col not in num_cols and col != "Churn"]

# %%
num_cols

# %%
cat_cols

# %%
data["Contract"].value_counts()

# %% [markdown]
# ## Feature Importance and Selection

# %%
# encode categorical columns
data_encoded = pd.get_dummies(data, columns=cat_cols)

# %%
for col in data_encoded.columns:
    print(col)

# %%
y = data_encoded["Churn"]
X = data_encoded.drop("Churn", axis=1)

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X.head()

# %%
# using RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# train the model
rf.fit(X_train, y_train)

# predict on test data
y_pred = rf.predict(X_test)

# check accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy", accuracy)

# check feature importance
print(rf.feature_importances_)

# %%
# create a dataframe of feature importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})

# sort the dataframe by feature importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# calculate the importance percentage
feature_importance_df['Percentage'] = (feature_importance_df['Importance'] / feature_importance_df['Importance'].sum()) * 100

# %%
plt.figure(figsize=(10, 8))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Features")
plt.ylabel("Importance")
# make the x-axis labels slanted
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# from the graph, the 1st 7 columns contribute the most to customer churn

# %%
print("Percentage contribution", feature_importance_df['Percentage'][:7].sum())
print()
print("Top 7 features:\n", "-"*15, sep="")
for col in feature_importance_df['Feature'][:7]:
    print(col)

# %% [markdown]
# From the output above, we can see that TotalCharges, MonthlyCharges, tenure, Contract, InternetService and PaymentMethod are the top 6 features that contribute the most to the churn.

# %%
# calculating the percentage contribution of top 6 features
# including all its sub-segments

contribution = feature_importance_df[feature_importance_df["Feature"].str.contains("Contract|InternetService|PaymentMethod")]["Percentage"].sum() + feature_importance_df["Percentage"][:3].sum()

print("Percentage contribution", contribution)

# %%
feature_imp_cols = feature_importance_df[feature_importance_df["Feature"].str.contains("Contract|InternetService|PaymentMethod")]["Feature"].to_list() + feature_importance_df["Feature"][:3].to_list()
feature_imp_cols

# %%
# define important columnsa
imp_cols = [
    "TotalCharges",
    "MonthlyCharges",
    "tenure",
    "Contract",
    "InternetService",
    "PaymentMethod",
]

# %% [markdown]
# ### Selected Features
# 1. TotalCharges
# 2. MonthlyCharges
# 3. tenure
# 4. Contract
# 5. InternetService
# 6. PaymentMethod

# %% [markdown]
# ## Exploratory Data Analysis

# %%
cat_cols

# %%
num_cols

# %% [markdown]
# ### Insights (Numerical Features)

# %%
# define subplot axes
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

num = 0 # track the axes position

for col in num_cols:
    sns.boxplot(x="Churn", y=col, data=data, ax=axes[num])
    axes[num].set_title(f"Boxplot of {col} vs Churn")
    num += 1

plt.tight_layout() # to avoid overlapping of subplots
plt.subplots_adjust(wspace=0.5) # space between subplots
plt.show()

# %% [markdown]
# #### Notes
# - **Tenure:** Majority of the customers who churned spent less months with the company. It goes to show that the longer the customer stays with the company, the lesser the chances of them churning.
# - **Monthly Charges:** Customers who churned were charged more on a monthly basis. This could be due to the fact that they were on a short term contract. The boxplot of the *tenure* feature shows that customers who churned were on a short term contract (possibly).
# - **Total Charges:** Customers who churned were charged relatively less than those who did not churn. However, there are outliers that showed that some customers who churned were charged more than those who did not churn.
# - Overall, **tenure**, **MonthlyCharges** and **TotalCharges** are important features that contribute to customer churn.

# %% [markdown]
# ### Insights (Categorical Features)

# %%
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))

x_num = 0 # track the x axis position
y_num = 0 # track the y axis position

for col in cat_cols:
    cross_tab = pd.crosstab(data['Churn'], data[col])
    cross_tab.plot(kind='bar', stacked=True, ax=axes[x_num, y_num], rot=0)
    axes[x_num, y_num].set_title(f"Churn vs {col}")

    # adjust the axes position
    if y_num == 3:
        x_num += 1
        y_num = 0
    else:
        y_num += 1

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Notes
# - Out of the customers who churned, the proportion of males and females are almost the same.
# - Whether or not a customer has a partner does not seem to have a significant impact on customer churn. The same applies to senior citizens and customers who have dependents.
# - Customers who have a 2yr contract with the company are less likely to churn. This is because they are tied to the company for 2 years. That trend continues with customers who have a 1yr or monthly based contract with the company.
# - Customers who have no internet service are less likely to churn. A significant portion of DSL customers did not churn. Fibre optic customers are more likely to churn (about 2x more likely).
# - Customers who pay via electronic check are more likely to churn. Customers who pay via mailed check are less likely to churn.
#
# The features that show high contrast with respect to customer churn are those whose opposite classes are far apart. From the graph the features that show high contrast are:
# 1. InternetService
# 2. Contract
# 3. PaymentMethod

# %% [markdown]
# ### Concluding Insights
#
# From the EDA, we can conclude that the following features are important in predicting customer churn:
# 1. Tenure
# 2. MonthlyCharges
# 3. TotalCharges
# 4. InternetService
# 5. Contract
# 6. PaymentMethod
#
# This corresponds to the features that were selected using the feature importance selection from the random forest classifier.

# %% [markdown]
# ## Training the Model (MLP)

# %%
xcols = feature_imp_cols
xcols

# %%
data_encoded.head()[xcols].info()

# %%
data_encoded[xcols].shape

# %%
data_encoded["Churn"].head()

# %%
data_encoded["Churn"] = pd.get_dummies(data_encoded["Churn"], drop_first=True)

# %%
# split the data into train, validation and test
X_train, X_test, y_train, y_test = train_test_split(data_encoded[xcols], data_encoded["Churn"], test_size=0.2, random_state=42)

# split the train data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# show the shape of train, validation and test data
print("Train data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)

# %%
X_train.head()

# %%
# scale the numerical columns

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# %%
# convert data type to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_val = y_val.astype('float32')
y_test = y_test.astype('float32')

# %%
X_train.shape

# %%
y_train.head()

# %% [markdown]
# ### Define Model Architecture

# %%
def create_model(optimizer='adam'):
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

    return model

# %%
# implement grid search using kerasClassifier
# perform grid search
model_kcf = KerasClassifier(build_fn=create_model, verbose=2)

# define the grid search parameters
batch_size = [20, 32, 64, 128]
epochs = [20, 40, 60, 80, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator=model_kcf, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# %%
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print(grid_result.best_estimator_)

# %%
grid_result.best_estimator_.optimizer

# %% [markdown]
# ### Retrain the Model with best parameters

# %%
churn_model = create_model(optimizer=grid_result.best_estimator_.optimizer)

history = churn_model.fit(X_train, y_train, batch_size=grid_result.best_params_["batch_size"], epochs=grid_result.best_params_["epochs"], verbose=2, validation_data=(X_val, y_val))

# %%
# Evaluate the data with the test set
test_loss, test_accuracy = churn_model.evaluate(X_test, y_test)
print("Fashion Model Test Accuracy: ", str(test_accuracy))

# %%
epochs = history.epoch
history = history.history

# %%
# Visualize the train and test losses
plt.title("Training Loss vs Validation Loss")
plt.plot(epochs, history["loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# %%
# Visualize Accuracy for train and validation sets
plt.title("Training Accuracy vs Validation Accuracy")
plt.plot(epochs, history["accuracy"], label="Train Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# %%
# check auc score
y_pred = churn_model.predict(X_test)
print("auc score:", roc_auc_score(y_test, y_pred))

# check accuracy
y_pred = churn_model.predict(X_test)
y_pred = np.round(y_pred)
print("accuracy_score", accuracy_score(y_test, y_pred))

# %% [markdown]
# #### Notes
# - Grid search has given us the optimal parameters for the MLP model.
# - The model has good accuracy, but the loss on the validation set is high. This could be due to overfitting.
# - One of the ways to solve the problem of overfitting is to use dropout layers. Dropout layers randomly drop some neurons during training.

# %%
# implement dropout model
def op_dropout_model(optimizer='adam'):
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

    return model

# %%
churn_model = op_dropout_model(optimizer=grid_result.best_estimator_.optimizer)

history = churn_model.fit(X_train, y_train, batch_size=grid_result.best_params_["batch_size"], epochs=grid_result.best_params_["epochs"], verbose=2, validation_data=(X_val, y_val))

# %%
# Evaluate the data with the test set
test_loss, test_accuracy = churn_model.evaluate(X_test, y_test)
print("Fashion Model Test Accuracy: ", str(test_accuracy))

# %%
epochs = history.epoch
history = history.history

# %%
# Visualize the train and test losses
plt.title("Training Loss vs Validation Loss")
plt.plot(epochs, history["loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# %%
# Visualize Accuracy for train and validation sets
plt.title("Training Accuracy vs Validation Accuracy")
plt.plot(epochs, history["accuracy"], label="Train Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# %%
# check auc score
y_pred = churn_model.predict(X_test)
print("auc score:", roc_auc_score(y_test, y_pred))

# check accuracy
y_pred = churn_model.predict(X_test)
y_pred = np.round(y_pred)
print("accuracy_score", accuracy_score(y_test, y_pred))

# %% [markdown]
# #### Notes
# - The loss on the validation set has reduced significantly. This shows that the model is no longer overfitting.
# - The accuracy on the validation set has also increased.
# - Another way to reduce overfitting is reducing the number of neurons and hidden layers.

# %%
# implement grid search using kerasClassifier
def op_red_layer_model(optimizer='adam'):
    input_layer = Input(shape=(X_train.shape[1],))
    dense_layer_1 = Dense(32, activation='relu')(input_layer)
    dense_layer_2 = Dense(16, activation='relu')(dense_layer_1)
    dense_layer_3 = Dense(8, activation='relu')(dense_layer_2)
    output_layer = Dense(1, activation='sigmoid')(dense_layer_3)

    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# %%
churn_model = op_red_layer_model(optimizer=grid_result.best_estimator_.optimizer)

history = churn_model.fit(X_train, y_train, batch_size=grid_result.best_params_["batch_size"], epochs=grid_result.best_params_["epochs"], verbose=2, validation_data=(X_val, y_val))

# %%
# Evaluate the data with the test set
test_loss, test_accuracy = churn_model.evaluate(X_test, y_test)
print("Fashion Model Test Accuracy: ", str(test_accuracy))

# %%
epochs = history.epoch
history = history.history

# %%
# Visualize the train and test losses
plt.title("Training Loss vs Validation Loss")
plt.plot(epochs, history["loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# %%
# Visualize Accuracy for train and validation sets
plt.title("Training Accuracy vs Validation Accuracy")
plt.plot(epochs, history["accuracy"], label="Train Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# %%
# check auc score
y_pred = churn_model.predict(X_test)
print("auc score:", roc_auc_score(y_test, y_pred))

# check accuracy
y_pred = churn_model.predict(X_test)
y_pred = np.round(y_pred)
print("accuracy_score", accuracy_score(y_test, y_pred))

# %% [markdown]
# ### Final Thoughts
#
# - The loss has significantly reduced and the accuracy has increased.
# - The model is ready to be deployed.

# %% [markdown]
# ## Save Model

# %%
# save the model
churn_model.save("../models/churn_model.h5")

# save the scaler
import joblib
joblib.dump(scaler, "../models/scaler.pkl")

# save the encoder
cols = [col for col in imp_cols if col not in num_cols] # get cols from imp_cols that are not in num_cols
joblib.dump(pd.get_dummies(data[cols]), "../models/encoder.pkl")

# %%
cols = [col for col in imp_cols if col not in num_cols]
cols



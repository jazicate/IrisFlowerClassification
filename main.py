import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate


# *****Import Dataset*****
try:
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(filepath_or_buffer='iris/iris.data', names=columns)
except ImportError:
    print("Error: Import Dataset failed")
    exit()
else:
    # Check to see if imported dataset
    df = pd.DataFrame(data)
    print("***Original Dataset***")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


# *****Preprocess Dataset*****
# Data cleaning/Check
print("\n\n***Data Checking***")
print(data.shape)
print(data.sample(5))
print(data.tail())
print("Any NaN values: Yes" if data.isna().values.any() else "Any NaN values: No")
print("Any Duplicate values: Yes" if data.duplicated().values.any() else "Any Duplicate values: No")
print(data.info())


# We will now use all three Iris types, no need for class mapping
df['class'] = LabelEncoder().fit_transform(df['class'])  # Encode the class column
print("\n***Preprocessed Dataset***")
df.reset_index(drop=True, inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)


# Transform: Prepare feature matrix X and output vector y
X = df.drop('class', axis=1)
y = df['class']


# Split data: training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# *****Feature Scaling (Standardization) for Logistic Regression*****
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# *****Visualize Preprocessed Data*****
sns.pairplot(df, hue='class', height=3)
plt.title("Pairplot of Iris Dataset")
plt.savefig('Data Visualization/Visualization of Preprocessed Dataset.png')


# *****Function to Evaluate Models and Return Metrics*****
def evaluate_model(model, X_set, y_set, set_label):
    results = {}
    print(f"\n\n***Evaluations for {set_label} Set****")

    y_predicted = model.predict(X_set)  # Use model's predicted labels (not probabilities)
    print(f"Actual Value:\n{y_set}")  # y
    print(f"Predicted Value: {y_predicted}\n")  # predicted y

    # Confusion Matrix
    c_m = confusion_matrix(y_set, y_predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(c_m, annot=True, cmap='Blues', fmt='d')
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    plt.title(f"Confusion Matrix for Iris Dataset {set_label} Set")
    plt.savefig(f'Data Visualization/Confusion Matrix for Iris Dataset {set_label} Set')

    # Accuracy
    accuracy = accuracy_score(y_set, y_predicted)
    print(f"{set_label} Set Accuracy: {accuracy}")
    results['Accuracy'] = accuracy

    # Error
    error = 1 - accuracy
    print(f"{set_label} Set Error: {error}")
    results['Error'] = error

    # Sensitivity (Recall)
    sensitivity = recall_score(y_set, y_predicted, average='macro')  # Changed to handle multi-class
    print(f"{set_label} Set Sensitivity: {sensitivity}")
    results['Sensitivity'] = sensitivity

    # Specificity
    tn, fp, fn, tp = c_m.ravel() if c_m.shape == (2, 2) else (0, 0, 0, 0)
    try:
        specificity = tn / (tn + fp)
    except ZeroDivisionError:
        specificity = 0
    print(f"{set_label} Set Specificity: {specificity}")
    results['Specificity'] = specificity

    # Precision
    precision = precision_score(y_set, y_predicted, average='macro')  # Changed to handle multi-class
    print(f"{set_label} Set Precision: {precision}")
    results['Precision'] = precision

    # F1-Score
    f1 = f1_score(y_set, y_predicted, average='macro')  # Changed to handle multi-class
    print(f"{set_label} Set F1 Score: {f1}")
    results['F1 Score'] = f1

    # Log-Loss
    y_probs = model.predict_proba(X_set)  # Get probabilities for all classes
    lloss = log_loss(y_set, y_probs)  # Use predicted probabilities for Log-Loss
    print(f"{set_label} Set Logarithmic Loss (Cost): {lloss:.3f}")
    results['Log-Loss'] = lloss

    # Area Under the ROC Curve (AUC)
    roc_auc_value = roc_auc_score(y_set, y_probs, multi_class='ovr')  # Use probabilities, not predicted labels
    print(f"{set_label} Set Area Under the ROC Curve: {roc_auc_value}")
    results['AUC'] = roc_auc_value

    # Return the results for tabulation
    return results


# Initialize dictionaries to store model evaluation results
results_nb = {}
results_lr = {}
results_nn = {}


# *****Naive Bayes Classifier*****
print("\n\n***Naive Bayes Model***")
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
results_nb['Training'] = evaluate_model(model_nb, X_train, y_train, set_label='Training')
results_nb['Test'] = evaluate_model(model_nb, X_test, y_test, set_label='Test')


# *****Logistic Regression (SGDClassifier) - Hyperparameter Tuning*****
print("\n\n***Logistic Regression Model Tuning***")
model_lr = SGDClassifier(loss='log_loss', random_state=42)

# Update the hyperparameter grid to ensure eta0 is positive
param_grid_lr = {
    'alpha': [0.0001, 0.001, 0.01],  # Regularization term
    'max_iter': [1000, 3000, 5000],  # Number of iterations
    'learning_rate': ['constant', 'optimal', 'invscaling'],  # Learning rate schedule
    'eta0': [0.01, 0.1, 0.001]  # Ensure eta0 is positive for 'constant' and 'invscaling' schedules
}

# Perform grid search with cross-validation
grid_search_lr = GridSearchCV(estimator=model_lr, param_grid=param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train_scaled, y_train)

# Get the best model from grid search
model_lr_best = grid_search_lr.best_estimator_

# Evaluate the best Logistic Regression model
results_lr['Training'] = evaluate_model(model_lr_best, X_train_scaled, y_train, set_label='Training')
results_lr['Test'] = evaluate_model(model_lr_best, X_test_scaled, y_test, set_label='Test')


# *****Neural Network (MLPClassifier)*****
print("\n\n***Neural Network Model***")
model_nn = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=3000, random_state=42)
model_nn.fit(X_train, y_train)
results_nn['Training'] = evaluate_model(model_nn, X_train, y_train, set_label='Training')
results_nn['Test'] = evaluate_model(model_nn, X_test, y_test, set_label='Test')


# Combine results into a table
headers = ["Metric", "Naive Bayes (Train)", "Naive Bayes (Test)",
        "Logistic Regression (Train)", "Logistic Regression (Test)",
        "Neural Network (Train)", "Neural Network (Test)"]


metrics = ['Accuracy', 'Error', 'Sensitivity', 'Specificity', 'Precision',
        'F1 Score', 'Log-Loss', 'AUC']


# Prepare the rows for the table
table_data = []
for metric in metrics:
    row = [metric]
    row.append(results_nb['Training'].get(metric, '-'))
    row.append(results_nb['Test'].get(metric, '-'))
    row.append(results_lr['Training'].get(metric, '-'))
    row.append(results_lr['Test'].get(metric, '-'))
    row.append(results_nn['Training'].get(metric, '-'))
    row.append(results_nn['Test'].get(metric, '-'))
    table_data.append(row)


# Print the results as a table
print("\n\n***Model Comparison Results***")
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

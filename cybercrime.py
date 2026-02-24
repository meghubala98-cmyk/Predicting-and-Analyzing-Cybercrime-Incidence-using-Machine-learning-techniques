import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import time

data = pd.read_csv('C:\\Users\\meghu\\OneDrive\\Desktop\\Meghna\'s Master Project\\twitter_racism_parsed_dataset.csv')
# Add these lines here to handle NaN values and ensure all data are strings
print("NaN values before:", data['Text'].isna().sum())
data = data.dropna(subset=['Text'])  # Remove NaNs
print("NaN values after:", data['Text'].isna().sum())
data['Text'] = data['Text'].astype(str)  # Ensure all data are strings
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Text'])

# Split the data into training and testing sets
y = data['oh_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()
# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set and evaluate accuracy
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))
print("Naive Bayes Precision:", nb_precision)
print("Naive Bayes Recall:", nb_recall)
print("Naive Bayes F1-Score:", nb_f1)
nb_time = time.time() - start_time
print("Naive Bayes time taken:", nb_time)

start_time = time.time()
# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set and evaluate accuracy
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1-Score:", rf_f1)
rf_time = time.time() - start_time
print("Random Forest time taken:", rf_time)

start_time = time.time()
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on your training data
knn_classifier.fit(X_train, y_train)

# Make predictions on your test data
knn_predictions = knn_classifier.predict(X_test)

# Evaluate the performance of the classifier
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions)

# Print the evaluation metrics
print("KNN Accuracy:", knn_accuracy)
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)
print("KNN F1-Score:", knn_f1)
knn_time = time.time() - start_time
print("KNN time taken:", knn_time)

start_time = time.time()
# Train a logistic regression classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

# Make predictions on the test set and evaluate performance
lr_predictions = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Logistic Regression Precision:", lr_precision)
print("Logistic Regression Recall:", lr_recall)
print("Logistic Regression F1-Score:", lr_f1)
lr_time = time.time() - start_time
print("Logistic Regression Time taken", lr_time)

start_time = time.time()
# Train a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set and evaluate performance
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)
print("Decision Trees Accuracy:", dt_accuracy)
print("Decision Trees Precision:", dt_precision)
print("Decision Trees Recall:", dt_recall)
print("Decision Tress F1-Score:", dt_f1)
dt_time = time.time() - start_time
print("Decision Tree time Taken:", dt_time)

start_time = time.time()
# Train a Support Vector Machines (SVM) classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set and evaluate performance
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM F1-Score:", svm_f1)
svm_time = time.time() - start_time
print("SVM time taken:", svm_time)

start_time = time.time()
# Train a Neural Network classifier
nn_classifier = MLPClassifier()
nn_classifier.fit(X_train, y_train)

# Make predictions on the test set and evaluate performance
nn_predictions = nn_classifier.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)
nn_precision = precision_score(y_test, nn_predictions)
nn_recall = recall_score(y_test, nn_predictions)
nn_f1 = f1_score(y_test, nn_predictions)
print("Neural Networks Accuracy:", nn_accuracy)
print("Neural Networks Precision:", nn_precision)
print("Neural Networks Recall:", nn_recall)
print("Neural Networks F1-Score:", nn_f1)
nn_time = time.time() - start_time
print("Neural Networks", nn_time)

start_time = time.time()
# Train an AdaBoost classifier
ab_classifier = AdaBoostClassifier()
ab_classifier.fit(X_train, y_train)

# Make predictions on the test set and evaluate performance
ab_predictions = ab_classifier.predict(X_test)
ab_accuracy = accuracy_score(y_test, ab_predictions)
ab_precision = precision_score(y_test, ab_predictions)
ab_recall = recall_score(y_test, ab_predictions) 
ab_f1 = f1_score(y_test, ab_predictions)
print("Adaboost Accuracy:", ab_accuracy)
print("AdaBoost Precision:", ab_precision)
print("AdaBoost Recall:", ab_recall)
print("AdaBoost F1-Score:", ab_f1)
ab_time = time.time() - start_time
print("AdaBoost Time Taken", ab_time)

# Plot the results using a grouped bar chart
labels = ['Naive Bayes', 'k-Nearest Neighbors', 'Random Forest','Decision Tree', 'Logistic Regression', 'SVM', 'Neural Network', 'AdaBoost']
accuracy = [nb_accuracy, knn_accuracy, rf_accuracy,dt_accuracy, lr_accuracy, svm_accuracy, nn_accuracy, ab_accuracy]
precision = [nb_precision, knn_precision, rf_precision,dt_precision, lr_precision, svm_precision, nn_precision, ab_precision]
recall = [nb_recall, knn_recall, rf_recall,dt_recall, lr_recall, svm_recall, nn_recall, ab_recall]
f1 = [nb_f1, knn_f1, rf_f1,dt_f1, lr_f1, svm_f1, nn_f1, ab_f1]

x = list(range(len(labels)))
width = 0.2

fig, ax = plt.subplots(figsize=(10,6))

rects1 = ax.bar(x, accuracy, width, label='Accuracy')
rects2 = ax.bar([i+width for i in x], precision, width, label='Precision')
rects3 = ax.bar([i+width*2 for i in x], recall, width, label='Recall')
rects4 = ax.bar([i+width*3 for i in x], f1, width, label='F1-score')

ax.set_xlabel('Result of 8 Different Classifiers')
ax.set_ylabel('Score')
ax.set_title('Different Metrics Performance Evaluation of 8 Classifiers')
ax.set_xticks([i+1.5*width for i in x])
ax.set_xticklabels(labels)
ax.legend()

plt.show()

def predict_cyberbullying_all(text, classifiers, classifier_names, vectorizer):
    # Transform the input text
    text_vector = vectorizer.transform([text])

    # Initialize a dictionary for the predictions
    predictions = {}

    # Iterate through classifiers and make predictions
    for i in range(len(classifiers)):
        classifier = classifiers[i]
        classifier_name = classifier_names[i]

        prediction = classifier.predict(text_vector)
        if prediction[0] == 1:
            predictions[classifier_name] = "Cyberbullying detected"
        else:
            predictions[classifier_name] = "No cyberbullying detected"

    return predictions

# Usage:
classifier_list = [nb_classifier, knn_classifier, rf_classifier, dt_classifier, lr_classifier, svm_classifier, nn_classifier, ab_classifier]
classifier_names = ['Naive Bayes', 'k-Nearest Neighbors', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'SVM', 'Neural Network', 'AdaBoost']

input_text = "@AAlwuhaib1977 Muslim mob violence against Hindus in Bangladesh continues in 2014. #Islam http://t.co/C1JBWJwuRc"
result = predict_cyberbullying_all(input_text, classifier_list, classifier_names, vectorizer)

for classifier_name, prediction in result.items():
    print(f'{classifier_name}: {prediction}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data: replace this with your own dataset
data = {
    'text': ['Sample text in English', 'Exemple de texte en français', 'தமிழில் மாதிரி உரை', 'Texto de muestra en español'],
    'language': ['English', 'French', 'Tamil', 'Spanish']
}

df = pd.DataFrame(data)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df['text'], df['language'], test_size=0.2, random_state=42)

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, train_labels)
svm_predictions = svm_classifier.predict(X_test)

# Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, train_labels)
nb_predictions = nb_classifier.predict(X_test)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, train_labels)
rf_predictions = rf_classifier.predict(X_test)

# Evaluate the models
def evaluate_model(predictions, true_labels, model_name):
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"{model_name} Classification Report:\n{report}\n")

evaluate_model(svm_predictions, test_labels, 'SVM')
evaluate_model(nb_predictions, test_labels, 'Naive Bayes')
evaluate_model(rf_predictions, test_labels, 'Random Forest')
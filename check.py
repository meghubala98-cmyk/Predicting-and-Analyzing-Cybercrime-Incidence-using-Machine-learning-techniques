# Load the dataset
data = pd.read_csv("C:\\Users\\meghu\\OneDrive\\Desktop\\Meghna's Master Project\\kaggle_parsed_dataset.csv")

# Preprocess the data
def preprocess(data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    data['text'] = data['text'].apply(lambda x: re.sub(r'\W+', ' ', x.lower()).split())
    data['text'] = data['text'].apply(lambda x: [word for word in x if word not in stop_words])
    data['text'] = data['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    data['text'] = data['text'].apply(lambda x: ' '.join(x))

    return data

data = preprocess(data)
threshold = 0.5
data['binary_label'] = (data['oh_label'] >= threshold).astype(int)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["Text"], data["binary_label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
def predict_cyberbullying(text):
    text_processed = preprocess(pd.DataFrame({"text": [text]}))["text"]
    text_tfidf = vectorizer.transform(text_processed)
    probability = clf.predict_proba(text_tfidf)[:, 1][0]
    return probability
sample_text = "This is a sample text."
probability = predict_cyberbullying(sample_text)
print(f"The probability of cyberbullying in the text is {probability:.2f}.")

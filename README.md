🚨 Phishing URL Detection using Machine Learning
(A Machine Learning model to classify URLs as Good or Bad using TF-IDF and Naive Bayes.)

📌 Project Overview

Phishing attacks are among the most common cyber security threats. Attackers often disguise malicious URLs as legitimate ones to trick users into providing sensitive information.

This project builds a machine learning-based classifier to detect malicious URLs using:

->TF-IDF Vectorization

->Multinomial Naive Bayes

->Confusion Matrix

->ROC Curve & AUC Score

->Accuracy evaluation

->Real-time URL prediction

The model learns patterns from a dataset of labeled URLs and predicts whether a new URL is good or bad.

📁 Dataset

The dataset used in this project (project2.csv) contains two columns:

Column	    Description

URL	         The URL string

Label	      good or bad classification

Example:

https://google.com          -good

http://freemoney-scams.ru   -bad

🛠️ Technologies Used

Python

Scikit-learn

Pandas

NumPy

Matplotlib

TF-IDF Vectorizer

Multinomial Naive Bayes

📘 Project Workflow
1. Load and Inspect Dataset
   
(The CSV file is loaded using pandas, and the first few rows are displayed for verification.)

3. Exploratory Data Analysis (EDA)
   
(A bar plot shows distribution of good vs bad URLs.
This helps identify dataset balance.)

5. Data Preprocessing
   
(Split data into training and testing sets (80/20).
Transform URL strings into numerical vectors using TF-IDF.0

7. Model Training
   
(We train a Multinomial Naive Bayes classifier, which works well for text-based features.)

9. Model Evaluation
    
Metrics used:

    Accuracy Score
   
    Confusion Matrix
   
    ROC Curve
   
    AUC Score
    
These help measure classification performance and false positives/negatives.

11. Real-Time URL Prediction

User can input any URL:

Enter the URL to test: http://example-test.com

Prediction: bad

📊 Visualization Examples
✔️ Class Distribution Plot
(Shows how many URLs are labeled good vs bad.)

✔️ Confusion Matrix
(Indicates how many URLs were correctly and incorrectly classified.0

✔️ ROC Curve
(Evaluates tradeoff between true positive and false positive rates.)

🧠 Code Snippet (Main Components)
(Full code is included inside the repository)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

📈 Results

The model achieved high accuracy depending on dataset quality.

ROC curve and AUC indicate strong discrimination capability.

Performs well for real-time URL classification.

🔮 Future Improvements

Add character-level features (URL length, digits, symbols)

Use advanced models (Random Forest, SVM, XGBoost)

Deploy as a web app using Flask or Streamlit

Build a browser extension for real-time phishing detection

📝 Conclusion

This project demonstrates a simple yet effective machine learning approach to phishing URL detection. Using TF-IDF and Naive Bayes, we can classify URLs with high accuracy and visualize model performance through various metrics.

The project is useful for:

Cybersecurity learning,
Text classification practice,
Real-world ML model deployment.

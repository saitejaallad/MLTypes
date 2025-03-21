Machine Learning (ML) is a field of artificial intelligence that enables computers to learn from data and make predictions without being explicitly programmed.
📌 Key Components of Machine Learning:
	1	Data – The information used to train ML models.
	2	Features – The input variables that influence predictions.
	3	Labels – The correct outputs in supervised learning.
	4	Algorithms – The mathematical models that learn from data.
	5	Training & Testing – The process of learning from past data and evaluating performance.

1. Supervised vs. Unsupervised Learning
1️⃣ Supervised Learning
	•	Definition: The model learns from labeled data, where the correct output is provided.
	•	Goal: Learn a mapping between input (X) and output (Y).
	•	Examples:
	◦	Predicting house prices (Regression)
	◦	Classifying emails as spam or not spam (Classification)
📌 Supervised Learning Algorithms: ✔️ Linear Regression – Predicting continuous values ✔️ Logistic Regression – Binary classification ✔️ Decision Trees, SVM, Naive Bayes – Classification models

2️⃣ Unsupervised Learning
	•	Definition: The model learns from unlabeled data (no correct outputs).
	•	Goal: Find hidden patterns or group similar data.
	•	Examples:
	◦	Customer segmentation in marketing
	◦	Anomaly detection in fraud detection
📌 Unsupervised Learning Algorithms: ✔️ Clustering – K-Means, Hierarchical Clustering ✔️ Dimensionality Reduction – PCA, t-SNE

✅ 2. Regression (Linear, Logistic)
Regression is used for predicting continuous values or classifying binary categories.
1️⃣ Linear Regression (For Continuous Values)
	•	Goal: Predict numerical values from input features.
	•	Example: Predicting house prices based on size.
📌 Formula:

Y=mX+b
Where:
	•	X   X = Input feature
	•	Y   Y = Output (prediction)
	•	m   m = Slope (weight)
	•	b   b = Intercept
🔹 Implementation in Python (Scikit-Learn):
python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)  # Train model
y_pred = model.predict(X_test)  # Predict values

2️⃣ Logistic Regression (For Classification)
	•	Goal: Classify data into two categories (Binary classification).
	•	Example: Predicting whether an email is spam or not (Yes/No).
	•	Uses Sigmoid Function to output probabilities:

🔹 Implementation in Python:
python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

✅ 3. Classification (Decision Trees, SVM, Naive Bayes)
Classification is used to categorize data into different groups.
1️⃣ Decision Trees
	•	A tree-like model that makes decisions by splitting the data.
	•	Used for both classification & regression.
📌 Example:
	•	Spam detection: If an email contains “$$$,” classify as spam.
🔹 Implementation:
python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

2️⃣ Support Vector Machine (SVM)
	•	Finds the best boundary (hyperplane) to separate different classes.
	•	Works well with high-dimensional data.
📌 Example:
	•	Face recognition – Separates different people’s faces.
🔹 Implementation:
python
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)

3️⃣ Naive Bayes
	•	Based on probability theory (Bayes' Theorem).
	•	Works well for text classification (e.g., spam filtering).

🔹 Implementation:
python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

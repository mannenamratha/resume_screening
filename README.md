
# Resume Classification using Machine Learning

## Overview
This project involves developing a machine learning model that classifies resumes into different job categories. It uses natural language processing (NLP) techniques to clean and transform resume data, and machine learning algorithms to predict job categories based on the resume content.

## Libraries Used
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **Matplotlib** & **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning algorithms and text preprocessing.
- **Spacy**: For natural language processing tasks.
- **Pickle**: For saving and loading trained machine learning models.
- **Google Colab**: For mounting Google Drive to load the dataset.

## Dataset
- The dataset used in this project is a CSV file containing resumes and corresponding job categories.
- Each resume is categorized into one of the following job categories:
  - Java Developer, Python Developer, HR, Data Science, etc.

## Steps Involved

### 1. Data Loading
The resume dataset is loaded from Google Drive using `pandas`.

```python
import pandas as pd
df = pd.read_csv(file_path)
```

### 2. Data Cleaning
A custom function `cleanResume()` is used to clean the resume text by removing URLs, special characters, and unnecessary symbols.

```python
def cleanResume(txt):
    # Removes URLs, special characters, and performs text cleaning
    # Returns cleaned text
```

### 3. Text Vectorization
The `TfidfVectorizer` from scikit-learn is used to convert the cleaned resume text into numerical features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")
text_vectorized = tfidf.fit_transform(df["Resume"])
```

### 4. Label Encoding
The job categories are encoded into numerical values using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
```

### 5. Model Training
- **K-Nearest Neighbors (KNN)** is used as the classifier. 
- The data is split into training and testing sets using `train_test_split`.
- The model is trained on the vectorized resume data.

```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
```

### 6. Prediction
- The model is used to predict the job category of a new resume.
- The predicted category is displayed based on the input resume.

```python
y_pred = clf.predict(X_test)
```

### 7. Model Evaluation
The model's performance is evaluated using the accuracy score.

```python
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_pred))
```

### 8. Saving the Model
The trained model and vectorizer are saved using `pickle` for future predictions.

```python
import pickle
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))
```

### 9. Resume Prediction Example
The project includes an example where a new resume text is input, cleaned, and classified using the trained model.

```python
cleaned_resume = cleanResume(myresume)
input_features = tfidf.transform([cleaned_resume])
category_name = category_mapping.get(prediction_id, "Unknown")
print("Predicted Category:", category_name)
```

## How to Run
1. Ensure all the required libraries are installed.
2. Load the dataset by providing the correct file path.
3. Run the script to clean the data, train the model, and make predictions.

## Result
The model achieves an accuracy of **98.44%** on the test data, accurately predicting the job categories of resumes.

## Future Improvements
- Use other classification algorithms to compare performance.
- Implement additional NLP techniques for better text preprocessing.
- Fine-tune the model for further improvement in accuracy.

---

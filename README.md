# spam_classifier
<br>
Author - Bishal Panthi & Varun Rayamajhi

<h1>Spam Email Classifier</h1>
This project implements a machine learning-based spam email classifier using Support Vector Machine (SVM) and the TF-IDF vectorization technique to detect spam emails. The classifier is trained on a dataset containing spam and non-spam (ham) emails and provides predictions on whether a given email is spam.

<h1>Project Overview</h1>
Email spam detection is an important application of machine learning and natural language processing (NLP). In this project, we build a spam classifier by extracting features from the text of emails using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and training an SVM classifier to distinguish between spam and non-spam emails.

<h1>Features of the Project</h1>
• Text Preprocessing: The email text data is preprocessed by removing special characters and numbers and lowercasing the text.
• TF-IDF Vectorization: The text is converted into numerical feature vectors using the TfidfVectorizer from sklearn.
• Support Vector Machine (SVM): An SVM model is trained to classify emails as either spam or not spam.
• Model Evaluation: The model's performance is evaluated using accuracy, root mean squared error (RMSE), and a confusion matrix.
• Interactive Prediction: You can input any email text and the model will predict if it is spam.

<h1>Table of Contents</h1>
→ Project Overview
→ Dataset
→ Installation
→ Usage
→ Model Evaluation
→ Further Improvements
→ Contributing
→ License


<h1>Dataset</h1>
The dataset used for this project is composed of emails labeled as spam or ham. It contains text data for each email and a corresponding label:

→ Label 1: The email is spam.
→ Label 0: The email is not spam (ham).
The email data undergoes the following preprocessing steps:

<h1>Removal of special characters and numbers.</h1>
Conversion to lowercase for normalization.
Application of TfidfVectorizer for feature extraction.
Installation
To run this project on your local machine, follow these steps:

1) Clone the repository:
git clone <repository-url>

2) Create a virtual environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3) Install the required dependencies:
pip install -r requirements.txt


Ensure that the following libraries are installed:

• numpy
• pandas
• scikit-learn

Usage
Once the project is set up, you can train the model and test it with new email data:

<h1>Training the Classifier</h1>
→ The project uses a TF-IDF vectorizer to transform the email data into numerical feature vectors and trains a Support Vector Classifier (SVC). The training process is outlined below:

<h1>Testing the Classifier</h1>
→ You can input a new email text and classify it as spam or not spam

<h1>Model Evaluation</h1>
The model is evaluated using several metrics, including:
        Accuracy: Measures the percentage of correctly classified emails.
        RMSE (Root Mean Squared Error): Provides an indication of prediction errors.
        Confusion Matrix: Shows the number of true positives, false positives, true negatives, and false negatives.

Further Improvements:

<h1> Several enhancements can be made to improve the spam classifier: </h1>

1) Hyperparameter Tuning: Use grid search or random search to find the best hyperparameters for the SVC model.
2) Ensemble Methods: Experiment with ensemble classifiers (e.g., Random Forest, XGBoost) to improve model accuracy.
3) Deep Learning Models: Explore advanced models such as LSTM or BERT for text classification.
4) Deploy the Model: Build a web app using Flask or Streamlit to deploy the model and allow users to input email texts for classification.    


<h1> License </h1>
This project is licensed under the MIT License.

</br>
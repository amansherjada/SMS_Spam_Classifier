# SMS Spam Classifier

This repository contains a machine learning project for classifying SMS messages as spam or not spam. The project includes a Streamlit web app for real-time prediction, a Jupyter notebook with data exploration and preprocessing, and various machine learning models for classification.

- **Model Details**: I utilized TF-IDF combined with Multinomial Naive Bayes (MNB) to achieve a perfect precision score of 1.0. This approach ensures that spam messages are accurately classified with minimal false positives.
- **Live Model**: You can interact with the spam classifier live on [Streamlit](https://amansmsfilter.streamlit.app/).
- **Dataset**: The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
- **Kaggle Notebook**: For a detailed exploration of the 10 ML models and their performance, visit my [Kaggle notebook](https://www.kaggle.com/code/amansherjadakhan/spam-classifier-exploring-10-ml-models).

## Repository Structure

- **`app.py`**: A Streamlit app for real-time SMS spam classification.
- **`model.pkl`**: Serialized machine learning model for prediction.
- **`requirements.txt`**: Python dependencies required for the project.
- **`spam.csv`**: Dataset used for training and evaluation.
- **`spam_sms_detection.ipynb`**: Jupyter notebook for data exploration, preprocessing, and model evaluation.
- **`vectorizer.pkl`**: Serialized TF-IDF vectorizer used for text feature extraction.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   ```

2. Navigate to the project directory:
   ```bash
   cd sms-spam-classifier
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the URL provided by Streamlit in your browser to access the app.

### Jupyter Notebook

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook spam_sms_detection.ipynb
   ```

2. Follow the instructions in the notebook for data exploration and model evaluation.

## Data Preprocessing

The dataset is cleaned and preprocessed by:
- Removing irrelevant columns.
- Encoding categorical target variables.
- Applying text transformations like lowercasing, tokenization, removing stop words, punctuation, and stemming.

## Models

The repository includes multiple machine learning models:
- Naive Bayes (Multinomial, Bernoulli)
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Trees (DT)
- Logistic Regression (LR)
- Random Forest (RF)
- AdaBoost
- Bagging
- Extra Trees
- Gradient Boosting
- XGBoost

The models are evaluated based on accuracy and precision, with a focus on precision due to the imbalanced nature of the dataset.

## Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) for machine learning tools.
- [Streamlit](https://streamlit.io/) for the interactive web app.
- [NLTK](https://www.nltk.org/) for text processing.

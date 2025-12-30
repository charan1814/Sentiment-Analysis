Sentiment Classification of IMDb Movie Reviews
Project Overview
This project focuses on performing sentiment classification on the IMDb Movie Review Dataset. The goal is to build and evaluate machine learning models capable of classifying movie reviews as either 'positive' or 'negative' based on their textual content. The process involves comprehensive text preprocessing, TF-IDF vectorization, and the training and evaluation of various classification algorithms.

Dataset
The dataset used is the IMDb Movie Review Dataset, containing 50,000 movie reviews along with their sentiment labels (positive/negative).

Methodology
1. Data Loading and Initial Exploration
The IMDb Dataset was loaded into a pandas DataFrame.
Initial exploration included viewing the first few rows and checking the shape of the dataset.
2. Text Preprocessing
To prepare the text data for machine learning models, the following steps were performed:

Tokenization: Each review was broken down into individual words.
Stopword Removal: Common English stopwords (e.g., 'the', 'is', 'a') were removed to focus on more meaningful terms.
Lemmatization: Words were reduced to their base or root form (e.g., 'running' -> 'run', 'better' -> 'good') using NLTK's WordNetLemmatizer.
3. TF-IDF Vectorization
The preprocessed (lemmatized) reviews were transformed into numerical feature vectors using TfidfVectorizer. This technique assigns weights to words based on their frequency in a document and their inverse frequency across the entire corpus, highlighting words that are significant to a specific review.
The feature matrix X was created with 50,000 samples and 5,000 features (the most relevant words).
4. Data Splitting
The vectorized data (X) and sentiment labels (y) were split into training and testing sets using train_test_split, with 80% for training and 20% for testing. This resulted in:
X_train: (40,000, 5,000)
X_test: (10,000, 5,000)
y_train: (40,000,)
y_test: (10,000,)
5. Model Training and Evaluation
Three different classification models from scikit-learn were trained and evaluated on the preprocessed data:

Logistic Regression
Multinomial Naive Bayes
Linear Support Vector Classifier (SVC)
Each model's performance was assessed using a classification report, providing metrics such as precision, recall, F1-score, and accuracy.

Results and Best Model
After training and evaluating all three models, the performance metrics were compared:

Logistic Regression: Accuracy 0.89, Macro Avg F1-score 0.89
Multinomial Naive Bayes: Accuracy 0.85, Macro Avg F1-score 0.85
Linear SVC: Accuracy 0.88, Macro Avg F1-score 0.88
The Logistic Regression model demonstrated the best overall performance, achieving the highest accuracy and F1-score of 0.89.

Classification Report for Best Model (Logistic Regression):
              precision    recall  f1-score   support

    negative       0.90      0.87      0.89      4961
    positive       0.88      0.90      0.89      5039

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
Interpretation: The Logistic Regression model shows strong and balanced performance across both negative and positive sentiment categories, indicating its effectiveness in correctly identifying both types of reviews.

Insights and Next Steps
The project successfully implemented a sentiment classification pipeline, identifying Logistic Regression as the top-performing model among the evaluated options.
Future Work: Consider hyperparameter tuning for the Logistic Regression model to potentially enhance its performance further.
Further Exploration: Investigate other advanced machine learning models (e.g., Random Forest, Gradient Boosting, or deep learning architectures) and ensemble methods to see if higher accuracy and F1-scores can be achieved for this sentiment classification task.

# Quora-question-checker

The Quora Question Similarity project is a machine learning model that aims to determine the similarity between two questions posted on Quora. It analyzes various features of the questions and uses a RandomForestClassifier and XGBClassifier to predict whether the questions are duplicates or not.

## Dataset

The project utilizes the Quora Question Pairs dataset, which contains pairs of questions along with their labels indicating if they are duplicates or not. The dataset is available in the `train.csv` file.

## Features

The following features are extracted and used for the prediction:

- Length of questions (`q1_len` and `q2_len`)
- Number of words in questions (`q1_num_words` and `q2_num_words`)
- Number of common words between questions (`word_common`)
- Total number of unique words in both questions (`word_total`)
- Word share between questions (`word_share`)


## Results

The RandomForestClassifier and XGBClassifier models are trained and evaluated using the extracted features. The accuracy score on the test set is displayed.

## Future Improvements

- Explore other machine learning algorithms and compare their performance.
- Implement cross-validation for more robust model evaluation.
- Experiment with additional features and feature engineering techniques.
- Fine-tune the model hyperparameters to improve performance.


## Acknowledgments

- This project is based on the Quora Question Pairs dataset.
- The RandomForestClassifier and XGBClassifier models are used from the scikit-learn and XGBoost libraries, respectively.
- The project serves as an example of building a question similarity model using machine learning techniques.

Feel free to contribute to the project by suggesting improvements or adding new features!




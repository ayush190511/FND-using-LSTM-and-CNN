# FND-using-LSTM+CNN-and-several-other-machine-learning-algorithms

## Research Objective and Scope
- The primary goal is to replicate a deep learning approach proposed in the research paper, "Fake news detection: A hybrid CNN-RNN based deep learning approach."

- A key objective is to compare the performance of this hybrid deep learning model against several traditional machine learning algorithms.

Link for the paper: https://www.sciencedirect.com/science/article/pii/S2667096820300070


## Datasets used: 
FAKES and ISOT

## Description:

### Project Goal
* The project aims to build and compare several machine learning and deep learning models for fake news detection using a dataset named "FA-KES" and "ISOT"

### Data Preparation and Preprocessing
* The project loads the `FA-KES-Dataset.csv` and `ISOT.csv` file into a pandas DataFrame.
* It focuses exclusively on the **`article_content`** column for text analysis, filling any missing values with an empty string.
* Text cleaning is performed using a custom function that:
    * Converts all text to lowercase.
    * Removes punctuation and digits.
    * Tokenizes the text and removes common English stop words.
    * Applies Porter Stemming to the remaining words.
* The cleaned `article_content` is used as the feature set (X), and the `labels` column serves as the target variable (y).
* The data is split into a training set and a testing set with a ratio of 80:20 (`test_size=0.2`).

### Model Implementation and Evaluation
The project trains and evaluates several models across two main categories:

#### Traditional Machine Learning Models
* A **TfidfVectorizer** is used to convert the text data into numerical features.
* A **Pipeline** is created for each model to first perform TF-IDF vectorization and then train the classifier.
* The following models are trained and their performance is measured:
    * Logistic Regression (LR)
    * Random Forest Classifier (RF)
    * Multinomial Naive Bayes (MNB)
    * Stochastic Gradient Descent (SGD)
    * K-Nearest Neighbors (KNNs)
    * Decision Tree (DT)
    * AdaBoost Classifier (AB)
* For each model, the notebook calculates and stores the **accuracy, precision, recall, and F1-score**. It also computes and stores data for the **ROC curve and AUC score**.

#### Deep Learning Models
* Pre-trained **GloVe embeddings** (`glove.6B.100d.txt`) are loaded to represent words as 100-dimensional vectors.
* The text data is **tokenized** and sequences are **padded** to a fixed length of 300 to prepare for deep learning models.
* An embedding matrix is created to map the vocabulary to the pre-trained GloVe vectors.
* The following deep learning models are built using Keras:
    * **CNN Only (without embedding)**: A 1D convolutional neural network model that takes the GloVe vectors as direct input.
    * **CNN Only (with embedding)**: A CNN model that includes an embedding layer initialized with the GloVe embedding matrix.
    * **RNN Only (LSTM) (without embedding)**: An LSTM model that takes the GloVe vectors as direct input.
    * **RNN Only (LSTM) (with embedding)**: An LSTM model that includes an embedding layer initialized with the GloVe embedding matrix.
    * **CNN + RNN Hybrid**: A sequential model combining a 1D CNN layer, a max pooling layer, and an LSTM layer.
* Each deep learning model is trained for 10 epochs. Their performance is evaluated using the same metrics (accuracy, precision, recall, F1-score, and AUC) as the traditional models.

### Results and Visualization
* A pandas DataFrame is created to display the performance metrics for all tested models.
* A bar chart visually compares the **Accuracy** and **F1-Score** of all the models.
* Individual **ROC curves** are plotted for each model to visualize their performance, with the AUC score displayed for each.


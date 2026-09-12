# Text Classification using BERT Embeddings and Logistic Regression

A text classification project that uses **BERT-base-uncased** to generate contextual text embeddings and **Logistic Regression** as the classification model.

Instead of fine-tuning BERT end-to-end, this project uses BERT as a **feature extractor**. The generated embeddings are then used as input features for Logistic Regression.

---

## 🚀 Project Overview

The project follows a two-stage NLP pipeline:

**Raw Text → BERT Tokenization → BERT-base-uncased → Text Embeddings → Logistic Regression → Predicted Class**

BERT is responsible for generating meaningful contextual representations of the input text, while Logistic Regression performs the final classification.

---

## 🧠 Models Used

### BERT-base-uncased

The project uses the **BERT-base-uncased** model from Hugging Face Transformers.

BERT generates contextual representations of the input text. The representation associated with the **[CLS] token** is used as the text embedding.

BERT-base produces a **768-dimensional embedding** for each input text.

### Logistic Regression

The extracted BERT embeddings are passed to a Logistic Regression classifier.

This provides a relatively simple and efficient classification layer while leveraging the strong language representations learned by BERT.

---

## 📁 Project Structure

The repository can be organized into the following structure:

* `data/` — Dataset files
* `notebooks/` — Jupyter notebooks for experimentation
* `src/` — Preprocessing, embedding extraction, training, and prediction modules
* `models/` — Saved trained models
* `requirements.txt` — Python dependencies
* `README.md` — Project documentation

---

## 🛠️ Technologies Used

* 🐍 Python
* 🔥 PyTorch
* 🤗 Hugging Face Transformers
* 🧠 BERT
* 📊 Scikit-learn
* 🐼 Pandas
* 🔢 NumPy
* 📓 Jupyter Notebook

---

## 📊 Dataset

The project requires a dataset containing:

* A **Headlines** containing the input text
* A **is_sarcastic** containing the target class

The dataset can be adapted for different text classification tasks depending on the use case.

---

## 🔄 Methodology

### 1. Data Preprocessing

The text dataset is prepared before being passed to BERT.

Typical preprocessing includes:

* Preparing text data
* Encoding target labels
* Splitting data into training and testing sets

### 2. Text Tokenization

The input text is tokenized using the BERT-base-uncased tokenizer.

The tokenizer converts natural language text into tokens and prepares them in the format required by BERT.

### 3. BERT Embedding Extraction

BERT is used as a feature extractor rather than being fine-tuned.

For each text sample, BERT generates a contextual representation. The **[CLS] token embedding** is used as the representation of the complete text.

Each text is therefore converted into a **768-dimensional numerical vector**.

### 4. Logistic Regression Classification

The generated BERT embeddings are used as input features for Logistic Regression.

The classifier learns the relationship between the BERT embeddings and the target labels.

### 5. Prediction

For unseen text, the same process is followed:

**New Text → BERT Embedding → Logistic Regression → Predicted Class**

---

## 📈 Evaluation

The classification model can be evaluated using common classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics help evaluate how well the model performs on unseen data.

---

## ⚡ Why BERT + Logistic Regression?

A traditional text classification pipeline often uses TF-IDF features with a machine learning classifier.

This project instead uses:

**BERT Embeddings + Logistic Regression**

### Advantages

* BERT captures contextual meaning in text.
* Better representation of word relationships compared with basic bag-of-words approaches.
* Logistic Regression is relatively fast and simple.
* BERT can be used as a reusable feature extractor.
* The approach requires less training than full BERT fine-tuning.

### Limitations

* BERT embedding extraction can be computationally expensive.
* BERT itself is not fine-tuned for the specific classification task.
* Full fine-tuning may achieve better performance on some datasets.
* Storing embeddings for large datasets can require significant memory.

---

## 🎯 Use Cases

This approach can be used for various NLP classification tasks, including:

* Sentiment Analysis
* Spam Detection
* News Classification
* Topic Classification
* Intent Classification
* Toxic Comment Detection
* Customer Review Classification
* Email Classification
* Document Classification

---

## 🔬 Possible Experiments

The BERT embeddings can be combined with different machine learning algorithms and compared.

Possible classifiers include:

* Logistic Regression
* Support Vector Machine
* Random Forest
* XGBoost
* Neural Networks

The project can also compare different approaches:

**TF-IDF + Logistic Regression**

versus

**BERT Embeddings + Logistic Regression**

versus

**Fine-tuned BERT**

This comparison can demonstrate the differences between traditional NLP, transformer-based feature extraction, and end-to-end transformer fine-tuning.

---

## 🔮 Future Improvements

Possible improvements include:

* Fine-tuning BERT for the classification task
* Experimenting with different embedding pooling strategies
* Hyperparameter tuning
* Handling class imbalance
* Using stratified cross-validation
* Comparing different transformer models
* Adding an inference API
* Creating a Streamlit web application
* Containerizing the application using Docker

---

## 📌 Key Concept

The main idea of this project is to combine the strengths of **BERT** and **Logistic Regression**.

**BERT acts as the feature extractor**, generating rich contextual embeddings from text.

**Logistic Regression acts as the classifier**, using those embeddings to predict the target class.

### Overall Pipeline

**Text → BERT-base-uncased → 768-Dimensional Embedding → Logistic Regression → Classification**

---

##  Conclusion

This project demonstrates an effective approach to **text classification using BERT embeddings and Logistic Regression**. The `bert-base-uncased` model is used to convert text headlines into meaningful contextual embeddings, while Logistic Regression is used as the final classifier.

The dataset is divided into training and testing sets using stratified sampling, and the generated BERT embeddings are used to train the classification model. The trained model is then evaluated using **accuracy and a classification report**.
The project also provides a prediction function that can classify new text as **"Sarcastic"** or **"Not Sarcastic"**, making the trained model useful for testing unseen headlines. The Logistic Regression model is saved as `sarcasm_classifier.pkl` for future use.
Overall, this project shows how **pre-trained transformer models such as BERT can be combined with traditional machine-learning algorithms** to build a practical text classification system.



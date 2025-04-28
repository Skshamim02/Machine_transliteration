# Assamese Roman Script Transliteration 

This project builds and trains a deep learning model for **transliterating Romanized Assamese words** into the native Assamese script using a **BiLSTM**-based sequence-to-sequence model with TensorFlow and Keras.

## 📂 Project Structure

* **Training**:
   * Load and clean dataset (JSON format).
   * Character-level tokenization of input and output.
   * Build a BiLSTM Encoder-Decoder model.
   * Train and save the model.
* **Inference**:
   * Predict Assamese script from Romanized input.
   * Calculate Test Accuracy and BLEU Score for evaluation.

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Install Required Libraries

Install all dependencies easily:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow keras pandas numpy tqdm datasets fasttext nltk
```

### 3. Prepare Your Dataset

Make sure you have the following files in your **Google Drive**:
* `asm_train.json`
* `asm_test.json`

Update file paths in the code if you are not using Google Colab.

## 🚀 How to Run the Code

### 1. **Mount Google Drive** (for Colab users):

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. **Load and Preprocess Data**:
   * Fix JSON formatting if needed.
   * Add start (`\t`) and end (`\n`) tokens in target sequences.

### 3. **Train the Model**:

```python
model.fit(
    [roman_padded, ass_padded],
    y_train_padded,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
)
```

### 4. **Save the Trained Model**:

```python
model.save("/content/drive/MyDrive/internship/mt_bilstm_1l.h5")
```

### 5. **Load the Model for Inference**:

```python
saved_model = tf.keras.models.load_model('/content/drive/MyDrive/internship/mt_bilstm_1l.h5')
```

### 6. **Transliterate a Word**:

```python
input_word = input("\nEnter a romanized word: ")
print(f"\nTransliterated word: {transliterate(input_word_in_ids_padded)}")
```

## 📈 Evaluation Metrics

* **Test Accuracy**: Evaluated based on how many words were correctly transliterated.
* **BLEU Score**: Measures the quality of transliteration by comparing predicted output to the ground truth.

## 📋 Example

```bash
Enter a romanized word: eti
Transliterated word: এত
```

## 📄 Requirements

See `requirements.txt`, or:
* tensorflow >= 2.8.0
* keras >= 2.8.0
* pandas
* numpy
* tqdm
* datasets
* fasttext
* nltk

Install using:

```bash
pip install -r requirements.txt
```

## 📝 Notes

* The model uses **character-level tokenization** (each character is a separate token).
* Designed specifically for Assamese script transliteration.
* Built mainly for educational, research, and practical applications in Indian language technology.

## 🧑‍💻 Author

* SK Shamim Aktar

## 📜 License

This project is licensed under the MIT License.

## 📢 Important!

* Adjust file paths if you are running **locally** and not in **Google Colab**.
* If you plan to use a GPU, consider installing `tensorflow-gpu` for better performance.

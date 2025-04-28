  Assamese Roman Script Transliteration (BiLSTM Model)
This project builds and trains a deep learning model for transliterating Romanized Assamese words into the native Assamese script using a BiLSTM-based sequence-to-sequence model with TensorFlow and Keras.

📂 Project Structure
Training:

Load and clean dataset (JSON format).

Character-level tokenization of input and output.

Build a BiLSTM Encoder-Decoder model.

Train and save the model.

Inference:

Predict Assamese script from Romanized input.

Calculate Test Accuracy and BLEU Score for evaluation.

🛠 Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
2. Install Required Libraries
Install all dependencies easily:

bash
Copy
Edit
pip install -r requirements.txt
Or install manually:

bash
Copy
Edit
pip install tensorflow keras pandas numpy tqdm datasets fasttext nltk
3. Prepare Your Dataset
Make sure you have the following files in your Google Drive:

asm_train.json

asm_test.json

Update file paths in the code if you are not using Google Colab.

🚀 How to Run the Code
Mount Google Drive (for Colab users):

python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')
Load and Preprocess Data:

Fix JSON formatting if needed.

Add start (\t) and end (\n) tokens in target sequences.

Train the Model:

python
Copy
Edit
model.fit(
    [roman_padded, ass_padded],
    y_train_padded,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
)
Save the Trained Model:

python
Copy
Edit
model.save("/content/drive/MyDrive/internship/mt_bilstm_1l.h5")
Load the Model for Inference:

python
Copy
Edit
saved_model = tf.keras.models.load_model('/content/drive/MyDrive/internship/mt_bilstm_1l.h5')
Transliterate a Word:

python
Copy
Edit
input_word = input("\nEnter a romanized word: ")
print(f"\nTransliterated word: {transliterate(input_word_in_ids_padded)}")
📈 Evaluation Metrics
Test Accuracy:
Evaluated based on how many words were correctly transliterated.

BLEU Score:
Measures the quality of transliteration by comparing predicted output to the ground truth.

📋 Example
bash
Copy
Edit
Enter a romanized word: eti
Transliterated word : এত
📄 Requirements
See requirements.txt, or:

tensorflow >= 2.8.0

keras >= 2.8.0

pandas

numpy

tqdm

datasets

fasttext

nltk

Install using:

bash
Copy
Edit
pip install -r requirements.txt
📝 Notes
The model uses character-level tokenization (each character is a separate token).

Designed specifically for Assamese script transliteration.

Built mainly for educational, research, and practical applications in Indian language technology.

🧑‍💻 Author
[Your Name Here]

📜 License
This project is licensed under the MIT License.

📢 Important!
Adjust file paths if you are running locally and not in Google Colab.

If you plan to use a GPU, consider installing tensorflow-gpu for better performance.



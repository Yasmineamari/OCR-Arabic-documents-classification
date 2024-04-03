import pytesseract
import fitz  # PyMuPDF for reading PDF files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import arabic_reshaper
import os
import csv

# Set up Tesseract OCR path (Update the path based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Flask App setup
app = Flask(__name__)

# PDF Text Extraction function using PyMuPDF (MuPDF)
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(pdf_file)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    return text

# Classification Model
class DocumentClassifier:
    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    def train(self, documents, labels):
        self.model.fit(documents, labels)

    def predict(self, document):
        return self.model.predict([document])[0]

# Load training data from CSV
with open('s_data.csv', 'r', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    fieldnames = [field.strip() for field in csv_reader.fieldnames]
    documents = [{"text": row[fieldnames[0]], "label": row[fieldnames[1]]} for row in csv_reader]

# Create and train the classifier
classifier = DocumentClassifier()
texts = [sample["text"] for sample in documents]
labels = [sample["label"] for sample in documents]
classifier.train(texts, labels)

# Flask route for home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            # Create the 'uploads' directory if it doesn't exist
            if not os.path.exists("uploads"):
                os.makedirs("uploads")

            # Save the uploaded file to the 'uploads' directory
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.abspath(os.path.join("uploads", filename))
            uploaded_file.save(file_path)

            # Extract text from the PDF
            extracted_texts = [extract_text_from_pdf(file_path)[::-1]]

            # Print extracted text for debugging
            print(f"Extracted Text: {extracted_texts}")
            # Use Arabic Reshaper to correct text order
            reshaped_text = arabic_reshaper.reshape(extracted_texts[0])
             # Print reshaped text for debugging
            print(f"Reshaped Text: {reshaped_text}")           
            # Predict document category
            categories = [classifier.predict(text) for text in extracted_texts]

            # Print predicted categories for debugging
            print(f"Predicted Categories: {categories}")

            # Render result page
            return render_template("result.html", category=categories[0], extracted_text=extracted_texts[0])

    # Render the upload page
    return render_template("index.html")

# Run the Flask app
if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred: {e}")

from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

# Load the BART abstractive summarization model
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Initialize summarization pipeline for extractive summarization
summarization_pipeline = pipeline(task="summarization", model="t5-base", tokenizer="t5-base")

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(tokens)

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    try:
        reference = [word_tokenize(reference)]
        candidate = word_tokenize(candidate)
        return sentence_bleu(reference, candidate)
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

# Function to calculate ROUGE scores
def calculate_rouge(reference, candidate):
    try:
        rouge = Rouge()
        scores = rouge.get_scores(candidate, reference)
        return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']
    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        return 0.0, 0.0, 0.0

# Function for TF-IDF extractive summarization
def tfidf_summarize(text, num_sentences=3):
    try:
        sentences = sent_tokenize(text)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).flatten()
        
        # Check if there are enough sentences for summarization
        if len(sentences) < num_sentences:
            num_sentences = len(sentences)
        
        top_sentences = sentence_scores.argsort()[-num_sentences:][::-1]
        
        # Use integer conversion to ensure valid indices
        top_sentences = [int(i) for i in top_sentences]
        
        return [sentences[i] for i in top_sentences]
    except Exception as e:
        print(f"Error performing TF-IDF summarization: {e}")
        return []

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Summarization route
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if request.method == 'POST':
            # Get the input text from the form
            input_text = request.form['text']

            if not input_text:
                return render_template('index.html', error="Input text is empty.")

            abstractive_summary = ""  # Initialize with a default value
            extractive_summary = ""  # Initialize with a default value

            # Abstractive Summarization
            inputs = bart_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = bart_model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
            abstractive_summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Extractive Summarization
            extractive_summary = " ".join([item['summary_text'] for item in summarization_pipeline(input_text)])

            # Reference summary (replace with actual reference summaries)
            reference_summary = "This is a reference summary."

            # BLEU and ROUGE scores
            bleu_score_abstractive = calculate_bleu(reference_summary, abstractive_summary)
            rouge_1_abstractive, rouge_2_abstractive, rouge_l_abstractive = calculate_rouge(reference_summary, abstractive_summary)

            bleu_score_extractive = calculate_bleu(reference_summary, extractive_summary)
            rouge_1_extractive, rouge_2_extractive, rouge_l_extractive = calculate_rouge(reference_summary, extractive_summary)

            # Display the original text, abstractive summary, extractive summary, and evaluation scores
            return render_template('result.html', input_text=input_text, abstractive_summary=abstractive_summary,
                                   extractive_summary=extractive_summary, bleu_score_abstractive=bleu_score_abstractive,
                                   rouge_1_abstractive=rouge_1_abstractive, rouge_2_abstractive=rouge_2_abstractive,
                                   rouge_l_abstractive=rouge_l_abstractive, bleu_score_extractive=bleu_score_extractive,
                                   rouge_1_extractive=rouge_1_extractive, rouge_2_extractive=rouge_2_extractive,
                                   rouge_l_extractive=rouge_l_extractive)

    except Exception as e:
        print(f"Error in summarization: {e}")
        return render_template('index.html', error="An error occurred during summarization.")

if __name__ == '__main__':
    app.run(debug=True, port=8080)

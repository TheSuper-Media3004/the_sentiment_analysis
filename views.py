from flask import Flask, Blueprint, render_template, request, jsonify
from decouple import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from goose3 import Goose
from werkzeug.utils import secure_filename
import numpy as np
import os
import mimetypes
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from collections import Counter

app = Flask(__name__)
views = Blueprint(__name__, "views")
HUGGINGFACE_CACHE_DIR = config("HUGGINGFACE_CACHE_DIR", '')
TORCH_CACHE_DIR = config("TORCH_CACHE_DIR", '')
os.environ['TORCH_HOME'] = TORCH_CACHE_DIR

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(
    SENTIMENT_MODEL, cache_dir=HUGGINGFACE_CACHE_DIR)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL, cache_dir=HUGGINGFACE_CACHE_DIR)

@views.route('/', methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        input_type = request.form.get("type")
        input_text = ''
        label = request.form.get("label")

        if input_type == "text":
            input_text = request.form.get("input")
        elif input_type == "url":
            url = request.form.get("input")
            g = Goose()
            article = g.extract(url=url)
            input_text = article.cleaned_text
        elif input_type == "media":
            file = request.files.get("input")
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.root_path, 'static', 'files', filename)
                file.save(file_path)
                input_text = process_files(file_path, label)

        if not input_text:
            return jsonify({"error": "No valid text found for analysis"}), 400

        sentiment_analysis = find_text_sentiment_analysis(input_text)
        return jsonify(sentiment_analysis)

def chunk_text(text, max_len=510):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_len:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def process_files(file_path, label):
    mime_type, encoding = mimetypes.guess_type(file_path)
    if not mime_type:
        return None
    file_type, subtype = mime_type.split('/', 1)
    
    if subtype == "csv":
        return extract_text_from_csv(file_path, label)
    return None

def extract_text_from_csv(file_path, label):
    try:
        df = pd.read_csv(file_path)
        if label not in df.columns:
            return None
        return " ".join(df[label].astype(str).dropna())
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def find_text_sentiment_analysis(text):
    chunks = chunk_text(text)
    sentiment_dicts = []

    for chunk in chunks:
        encoded_text = tokenizer(chunk, return_tensors="pt")
        output = sentiment_model(**encoded_text)
        scores = softmax(output[0][0].detach().numpy())
        sentiment_dict = {
            'score_negative': scores[0],
            'score_neutral': scores[1],
            'score_positive': scores[2],
            'prominent_sentiment': ("NEGATIVE" if scores[0] > scores[1] and scores[0] > scores[2] else
                                    "POSITIVE" if scores[2] > scores[0] and scores[2] > scores[1] else
                                    "NEUTRAL")
        }
        sentiment_dicts.append(sentiment_dict)

    avg_sentiment_dict = {
        'score_negative': float(np.mean([d['score_negative'] for d in sentiment_dicts])),
        'score_neutral': float(np.mean([d['score_neutral'] for d in sentiment_dicts])),
        'score_positive': float(np.mean([d['score_positive'] for d in sentiment_dicts])),
        'prominent_sentiment': max(set([d['prominent_sentiment'] for d in sentiment_dicts]),
                                   key=[d['prominent_sentiment'] for d in sentiment_dicts].count)
    }

    return avg_sentiment_dict

app.register_blueprint(views)



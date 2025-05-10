from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import demoji

from utils import final_clean, AttentionLayer

app = Flask(__name__)

# — demoji emoji veritabanını indir (ilk çalıştırmada internet bağlantısı gerektirir) —
demoji.download_codes()

MODEL_PATH     = "gruudene_attention_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAXLEN         = 100

# Modeli custom layer ile yükle
model = load_model(
    MODEL_PATH,
    custom_objects={'AttentionLayer': AttentionLayer}
)

# Tokenizer’ı yükle
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/charts", methods=["GET"])
def charts():
    return render_template("charts.html")

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json(silent=True) or {}
    raw_text = data.get("text", "")

    # 1) Temizle
    clean   = final_clean(raw_text)
    # 2) Tokenize & pad
    seq     = tokenizer.texts_to_sequences([clean])
    pad_seq = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    # 3) Tahmin al
    preds = model.predict(pad_seq)
    idx   = int(preds.argmax(axis=1)[0])

    labels = [
        "age",
        "ethnicity",
        "gender",
        "not_cyberbullying",
        "other_cyberbullying",
        "religion"
    ]

    return jsonify({
        "label_index": idx,
        "label_name":  labels[idx],
        "confidence":  float(preds[0][idx])
    })

if __name__ == "__main__":
    app.run(debug=True)

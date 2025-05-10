# utils.py

import re
import string

import demoji
import contractions
from langdetect import detect, LangDetectException
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tensorflow as tf

# ——— NLTK Setup ———
nltk.download('wordnet',    quiet=True)
nltk.download('stopwords',  quiet=True)

stop_words   = set(stopwords.words('english'))
lemmatizer   = WordNetLemmatizer()


def remove_emoji(text: str) -> str:
    return demoji.replace(text, '')


def expand_contractions(text: str) -> str:
    return contractions.fix(text)


def remove_non_english(text: str) -> str:
    try:
        return text if detect(text) == "en" else ""
    except LangDetectException:
        return ""


def remove_all_entities(text: str) -> str:
    text = re.sub(r'[\r\n]+', ' ', text.lower())
    text = re.sub(r'(?:@|https?://)\S+', '', text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(w for w in text.split() if w not in stop_words)


def lemmatize(text: str) -> str:
    return ' '.join(lemmatizer.lemmatize(w) for w in text.split())


def remove_short_words(text: str, min_len: int = 2) -> str:
    return ' '.join(w for w in text.split() if len(w) >= min_len)


def correct_elongated_words(text: str) -> str:
    return re.sub(r'\b(\w+?)((\w)\3{2,})(\w*)\b', r'\1\3\4', text)


def final_clean(text: str) -> str:
    text = remove_emoji(text)
    text = expand_contractions(text)
    text = remove_non_english(text)
    text = remove_all_entities(text)
    text = lemmatize(text)
    text = remove_short_words(text)
    text = correct_elongated_words(text)
    return ' '.join(text.split())

# utils.py içindeki AttentionLayer tanımı

import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # trainable, dtype, name vb. tüm keyword argümanları alıp base sınıfa iletmiş oluyoruz
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs: [batch, timesteps, hidden_size]
        score = tf.nn.softmax(
            tf.reduce_sum(inputs, axis=2, keepdims=True),
            axis=1
        )
        context = inputs * score
        return tf.reduce_sum(context, axis=1)

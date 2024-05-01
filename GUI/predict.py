import pandas as pd
from nltk.tokenize import word_tokenize
from keras.models import load_model
import tensorflow as tf
from keras import layers
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
import gradio as gr
import time

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
validation = pd.read_csv("./validation.csv")

more_stopwords = {'u', "im", "day"}
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(more_stopwords)


# Basic text cleaning
def strip_html(text):
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text()


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = strip_html(text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def preprocess_text(text):
    text = clean_text(text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


train['tokens'] = train['text'].apply(preprocess_text)
test['tokens'] = test['text'].apply(preprocess_text)
validation['tokens'] = validation['text'].apply(preprocess_text)
all_tokens = pd.concat([train['tokens'], test['tokens'], validation['tokens']], axis=0)

model_w2v = Word2Vec(sentences=all_tokens, vector_size=100, window=10, sg=1, hs=1, min_count=1, workers=4)


def tokens_to_vectors(tokens, model):
    vectors = [model.wv[word] if word in model.wv else np.zeros((model.vector_size,)) for word in tokens]
    return np.array(vectors)


train['vectors'] = train['tokens'].apply(lambda tokens: tokens_to_vectors(tokens, model_w2v))
test['vectors'] = test['tokens'].apply(lambda tokens: tokens_to_vectors(tokens, model_w2v))
validation['vectors'] = validation['tokens'].apply(lambda tokens: tokens_to_vectors(tokens, model_w2v))
max_seq_length = max(train['vectors'].apply(len).max(), test['vectors'].apply(len).max(),
                     validation['vectors'].apply(len).max())

le = LabelEncoder()
train['label'] = le.fit_transform(train['sentiment'])
test['label'] = le.transform(test['sentiment'])
validation['label'] = le.transform(validation['sentiment'])


class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            raise ValueError("embedding dimension must be divisible by number of heads ")
        self.head_dim = embed_dim // num_heads

        self.query = Dense(embed_dim)
        self.key = Dense(embed_dim)
        self.value = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Linearly project the queries, keys, and values
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        # Split into multiple heads (batch_size, num_heads, max_seq_length, head_dim)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Calculate attention scores
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        # Linearly combine the heads
        output = self.combine_heads(concat_attention)
        return output, weights


posText = train[train['sentiment'] == 'positive']['text']
negText = train[train['sentiment'] == 'negative']['text']
neuText = train[train['sentiment'] == 'neutral']['text']

# Tokenize and extract words for each sentiment category
posWord = [word.lower() for text in posText for word in word_tokenize(str(text))]
negWord = [word.lower() for text in negText for word in word_tokenize(str(text))]
neuWord = [word.lower() for text in neuText for word in word_tokenize(str(text))]

sentiment_lexicon = {
    "positive": posWord,
    "negative": negWord,
    "neutral": neuWord
}


def get_average_vector(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Calculate average vectors for sentiment categories
average_vectors = {sentiment: get_average_vector(words, model_w2v) for sentiment, words in sentiment_lexicon.items()}

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")


    def user(user_message, history):
        return "", history + [[user_message, None]]


    def bot(history):
        print("Loading model...")
        model = load_model('cnn_model.keras', custom_objects={'MultiHeadAttention': MultiHeadAttention})
        text = history[-1][0]
        if text.replace(" ", "") == "":
            bot_message = "The empty comment is  not allowed. Please type a sentence."
        else:
            print("Input text ：", text)
            user_tokens = preprocess_text(text)
            user_vectors = get_average_vector(user_tokens, model_w2v).reshape(1, -1)
            padded_user_input = pad_sequences([user_vectors], maxlen=max_seq_length, dtype='float32', padding='post')
            prediction = model.predict(padded_user_input)
            predicted_label = np.argmax(prediction[0], axis=-1)
            pred_label = le.inverse_transform([predicted_label])
            bot_message = f"{pred_label}. Probability:{prediction[0][predicted_label]}"
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()

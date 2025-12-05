import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load IMDB word index
word_index = imdb.get_word_index()

# Reverse lookup dictionary
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model('simple_rnn_imdb.keras')


# Decode integer review back to text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


# Preprocess text input
def preprocess_text(text):

    if not text.strip():
        return sequence.pad_sequences([[2]], maxlen=500)

    words = text.lower().split()
    
    encoded_review = [word_index.get(word, 2) + 3 for word in words]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


# Streamlit UI
import streamlit as st

st.title("IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review to know its sentiment")

user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input = preprocess_text(user_input)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else "Negative"

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]:.4f}')

else:
    st.write('Please enter a movie review.')

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved models
encoder_model = tf.keras.models.load_model('encoder_model.keras')
decoder_model = tf.keras.models.load_model('decoder_model.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to generate clean HTML/CSS code
def generate_clean_code(ugly_code, tokenizer, max_sequence_length):
    # Convert the input code (ugly HTML) to a sequence of tokens
    input_sequence = tokenizer.texts_to_sequences([ugly_code])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')
    
    # Encode the input sequence to get the initial states for the decoder
    encoder_states = encoder_model.predict(input_sequence)
    
    # Start with the <start> token (if you have one defined in your tokenizer)
    start_token = tokenizer.word_index['<start>']  # Assuming <start> is in your vocabulary
    decoded_sequence = [start_token]
    
    # Initialize the states for the decoder
    states = encoder_states
    
    # Generate tokens one by one
    for _ in range(max_sequence_length):
        # Prepare the decoder input (the sequence generated so far)
        decoder_input = pad_sequences([decoded_sequence], maxlen=max_sequence_length, padding='post')
        
        # Predict the next token and the new states
        predictions, state_h, state_c = decoder_model.predict([decoder_input] + states)
        
        # Get the token with the highest probability
        predicted_token = np.argmax(predictions[0, -1, :])
        
        # Append the predicted token to the sequence
        decoded_sequence.append(predicted_token)
        
        # If the model predicts the <end> token, stop generating
        if predicted_token == tokenizer.word_index['<end>']:
            break
        
        # Update the states
        states = [state_h, state_c]
    
    # Convert the token sequence back to text (clean HTML)
    clean_code = tokenizer.sequences_to_texts([decoded_sequence])[0]
    
    return clean_code

# Example usage
ugly_code = '<div style="font-size: 18px; position: absolute">lorem ip some doller</div>'  # Example ugly code

# Assuming you've saved max_sequence_length during training
max_sequence_length = 51  # Example value; replace with your actual value

clean_code = generate_clean_code(ugly_code, tokenizer, max_sequence_length)
print(clean_code)

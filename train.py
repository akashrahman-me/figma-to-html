import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Example data
data = [
    [
        "<div style=\"display: flex; gap: 12px; align-items: start; justify-content: start\" > <div style=\"flex-shrink: 0; display: flex; align-items: center\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div style=\"display: block\"> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>",
        "<div style=\"display: flex; gap: 12px;\" > <div style=\"flex-shrink: 0;\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>"
    ],
    [
        "<div style=\"display: flex; gap: 12px; align-items: start; justify-content: start\" > <div style=\"flex-shrink: 0; display: flex; align-items: center\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div style=\"display: block\"> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>",
        "<div style=\"display: flex; gap: 12px;\" > <div style=\"flex-shrink: 0;\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>"
    ],
    [
        "<div style=\"display: flex; gap: 12px; align-items: start; justify-content: start\" > <div style=\"flex-shrink: 0; display: flex; align-items: center\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div style=\"display: block\"> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>",
        "<div style=\"display: flex; gap: 12px;\" > <div style=\"flex-shrink: 0;\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>"
    ],
    [
        "<div style=\"display: flex; gap: 12px; align-items: start; justify-content: start\" > <div style=\"flex-shrink: 0; display: flex; align-items: center\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div style=\"display: block\"> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>",
        "<div style=\"display: flex; gap: 12px;\" > <div style=\"flex-shrink: 0;\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>"
    ],
    [
        "<div style=\"display: flex; gap: 12px; align-items: start; justify-content: start\" > <div style=\"flex-shrink: 0; display: flex; align-items: center\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div style=\"display: block\"> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>",
        "<div style=\"display: flex; gap: 12px;\" > <div style=\"flex-shrink: 0;\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>"
    ],
    [
        "<div style=\"display: flex; gap: 12px; align-items: start; justify-content: start\" > <div style=\"flex-shrink: 0; display: flex; align-items: center\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div style=\"display: block\"> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>",
        "<div style=\"display: flex; gap: 12px;\" > <div style=\"flex-shrink: 0;\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>"
    ],
    [
        "<div style=\"display: flex; gap: 12px; align-items: start; justify-content: start\" > <div style=\"flex-shrink: 0; display: flex; align-items: center\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div style=\"display: block\"> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>",
        "<div style=\"display: flex; gap: 12px;\" > <div style=\"flex-shrink: 0;\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>"
    ],
    [
        "<div style=\"display: flex; gap: 12px; align-items: start; justify-content: start\" > <div style=\"flex-shrink: 0; display: flex; align-items: center\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div style=\"display: block\"> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>",
        "<div style=\"display: flex; gap: 12px;\" > <div style=\"flex-shrink: 0;\"> <img style=\"border-radius: 50%\" src=\"https://via.placeholder.com/80x80\" alt=\"\" /> </div> <div> <b style=\"font-weight: 500; font-size: 24px\">Akash Rahman</b> <p style=\"font-size: 18px\"> Lorem ipsum, dolor sit amet consectetur adipisicing elit. Veniam dolorem totam perferendis. </p> </div> </div>"
    ],
]

# Prepare the tokenizer to convert text to sequences of tokens
tokenizer = Tokenizer(filters='', lower=False)  # Keep case and all characters
texts = ['<start> ' + pair[0] + ' <end>' for pair in data] + ['<start> ' + pair[1] + ' <end>' for pair in data]
tokenizer.fit_on_texts(texts)

# Convert texts to sequences of token ids
input_sequences = tokenizer.texts_to_sequences([pair[0] for pair in data])
output_sequences = tokenizer.texts_to_sequences([pair[1] for pair in data])

# Define constants
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding token
max_sequence_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in output_sequences))
embedding_dim = 256
lstm_units = 512                         

# Pad sequences to ensure equal length
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

# Build the encoder
def build_encoder(vocab_size, embedding_dim, lstm_units, max_sequence_length):
    encoder_input = Input(shape=(max_sequence_length,), name="encoder_input")
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="encoder_embedding")(encoder_input)
    encoder_lstm, state_h, state_c = LSTM(lstm_units, return_state=True, name="encoder_lstm")(encoder_embedding)
    encoder_states = [state_h, state_c]  # Return the final states to initialize the decoder
    encoder_model = Model(encoder_input, encoder_states)
    return encoder_model

# Build the decoder
def build_decoder(vocab_size, embedding_dim, lstm_units, max_sequence_length):
    decoder_input = Input(shape=(max_sequence_length,), name="decoder_input")
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="decoder_embedding")(decoder_input)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_dense = Dense(vocab_size, activation='softmax', name="decoder_dense")
    
    # Inputs to the decoder
    decoder_state_input_h = Input(shape=(lstm_units,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(lstm_units,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # The decoder outputs
    decoder_lstm_output, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_outputs = decoder_dense(decoder_lstm_output)
    decoder_states = [state_h, state_c]
    
    # Define the decoder model
    decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return decoder_model

# Build the full seq2seq model
def build_seq2seq_model(encoder_model, decoder_model, max_sequence_length):
    encoder_input = encoder_model.input
    encoder_states = encoder_model.output  # Encoder output (states)
    
    # The decoder takes the encoder's states as initial states
    decoder_input = decoder_model.input[0]  # The input sequence to the decoder
    decoder_states_inputs = decoder_model.input[1:]  # The initial states (from the encoder)
    
    # Get the decoder's output and final states
    decoder_outputs, state_h, state_c = decoder_model([decoder_input] + encoder_states)
    
    # Define the full model
    seq2seq_model = Model([encoder_input, decoder_input], decoder_outputs)
    return seq2seq_model

# Build encoder, decoder, and the full seq2seq model
encoder_model = build_encoder(vocab_size, embedding_dim, lstm_units, max_sequence_length)
decoder_model = build_decoder(vocab_size, embedding_dim, lstm_units, max_sequence_length)
seq2seq_model = build_seq2seq_model(encoder_model, decoder_model, max_sequence_length)

# Compile the model
seq2seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare the decoder's target data by shifting the output sequence by one time step
decoder_target_data = np.expand_dims(output_sequences, -1)

# Train the model
seq2seq_model.fit([input_sequences, output_sequences], decoder_target_data, batch_size=64, epochs=10)

# Print the model summary
seq2seq_model.summary()

# Save the entire seq2seq model
seq2seq_model.save('seq2seq_model.keras')

# You can also save the encoder and decoder separately if needed
encoder_model.save('encoder_model.keras')
decoder_model.save('decoder_model.keras')


# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Max sequence length: {max_sequence_length}")

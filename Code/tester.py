from transformers import BertConfig, TFBertModel
import tensorflow as tf
# Building the config
config = BertConfig()

# Building the model from the config
model = TFBertModel(config)
sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]


model_inputs = tf.constant(encoded_sequences)
output = model(model_inputs)
print(output)
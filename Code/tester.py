from transformers import BertConfig, TFBertModel
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TFAutoModel
from transformers import AutoTokenizer

from transformers import TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "This is not what I want",
    "Thanks",
    "Correct",
    "Noooooooo",
    "Retry prompt",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(inputs)
predictions = tf.math.softmax(outputs.logits, axis=-1)
print([ 'neg' if max(i) == i[0] else 'pos' for i in predictions])
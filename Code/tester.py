from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import os

def model_create():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    # Original inputs including "Retry"
    raw_inputs = [
        "This is not what I want",
        "Thanks",
        "Correct",
        "Noooooooo",
        "Retry",
        "Retry"  # Adding "Retry" as a negative example
    ]

    # Fine-tuning data including "Retry" as a negative example
    fine_tuning_data = [
        ("This is not what I want", 0),  # Add more examples with correct sentiment labels
        ("Thanks", 1),
        ("Correct", 1),
        ("Noooooooo", 0),
        ("Retry", 0),
        ("Retry", 0)  # Adding "Retry" as a negative example
    ]


    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    batch_encoding = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
    inputs = (batch_encoding["input_ids"], batch_encoding["attention_mask"])

    # Process fine-tuning data
    fine_tuning_texts = [item[0] for item in fine_tuning_data]
    fine_tuning_labels = tf.convert_to_tensor([item[1] for item in fine_tuning_data])

    tokenizer_ft = AutoTokenizer.from_pretrained(checkpoint)
    inputs_ft = tokenizer_ft(fine_tuning_texts, padding=True, truncation=True, return_tensors="tf")

    # Extract tensors from the BatchEncoding object
    inputs_ft = (inputs_ft["input_ids"], inputs_ft["attention_mask"])

    # Fine-tune the model with additional data
    model_ft = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_ft.compile(optimizer=optimizer, loss=loss)
    model_ft.fit(inputs_ft, fine_tuning_labels, epochs=3)  # Train for a few epochs with additional data

    # Make predictions with the fine-tuned model
    outputs = model_ft(inputs)
    predictions = tf.math.softmax(outputs.logits, axis=-1)
    sentiments = ['neg' if max(i) == i[0] else 'pos' for i in predictions]
    print(sentiments)
    return model_ft

def model_use(raw_input, model):
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    batch_encoding = tokenizer(raw_input, padding=True, truncation=True, return_tensors="tf")
    inputs = (batch_encoding["input_ids"], batch_encoding["attention_mask"])
    outputs = model(inputs)
    predictions = tf.math.softmax(outputs.logits, axis=-1)
    print(['neg' if max(i) == i[0] else 'pos' for i in predictions])
    return


# Check if the 'model' folder exists and contains a model
model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")
model_path = os.path.join(model_folder, "my_model")

if os.path.exists(model_folder):
    # Load the existing model
     model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
     model.load_weights(model_path)
     print("Model loaded successfully.")

else:
    model = model_create()  # Assuming model_create() creates a new model instance
    model.save_weights(model_path)
    print("Model saved successfully.")

model_use("That person was mean.", model)
model_use("Screw off", model)
model_use("This is what I wanted", model)
model_use("Die", model)
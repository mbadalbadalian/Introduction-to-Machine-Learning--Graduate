from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import os
import warnings
import matplotlib.pyplot as plt

# To suppress all warnings
warnings.filterwarnings("ignore")

def model_create():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    # Original inputs including "Retry"
    raw_inputs = [
        "This is not what I wanted here",
        "Thanks a lot",
        "Correct answer",
        "Nooooooooooo",
        "Repeat that step",
        "Nice work",
        "Redo everything",
        "I rejoice in the correctness of the answer"
        "Retry this algorithm"  
    ]

    # Fine-tuning data including "Retry" as a negative example
    fine_tuning_data = [
        ("This is not what I want", 0),  # Add more examples with correct sentiment labels
        ("Thanks", 1),
        ("Correct", 1),
        ("No", 0),
        ("Repeat",0),
        ("Nice",1),
        ("This is it!",1),
        ("Redo", 0),
        ("I rejoice in the answer",1),
        ("Retry", 0)  # Adding "Retry" as a negative example
    ]
    fine_tuning_data.extend([
    ("I'm not happy with this", 0),
    ("Terrible", 0),
    ("Unsatisfied", 0),
    ("Displeased", 0),
    ("That was awful", 0),
    ("I'm dissatisfied", 0),
    ("Worst", 0),
    ("This is not right", 0),
    ("I'm disappointed", 0),
    ("Not good", 0),
    ("I hate it", 0),
    ("I dislike it", 0),
    ("This is a failure", 0),
    ("I'm not content", 0),
    ("This is frustrating", 0),
    ("Awful", 0),
    ("Very bad", 0),
    ("Not pleased", 0),
    ("I'm angry", 0),
    ("This is terrible", 0),
    ("I'm not satisfied", 0),
    ("I'm not happy", 0),
    ("This is unacceptable", 0),
    ("That was a mistake", 0),
    ("I'm furious", 0),
    ("This is wrong", 0),
    ("Disappointed", 0),
    ("Redo the algorithm",0),
    ("I'm not okay with this", 0),
    ("I'm upset", 0),
    ("This is not satisfactory", 0),
    ("I'm not pleased with this", 0),
    ("This is not acceptable", 0),
    ("I'm not content with this", 0),
    ("This is awful", 0),
    ("That's not good", 0),
    ("This is not what I wanted", 0),
    ("This is not what I expected", 0),
    ("I'm not happy with the answer", 0),
    ("I'm not satisfied with the answer", 0),
    ("This is not the answer I wanted", 0),
    ("I expected a better answer", 0),
    ("This is not a good answer", 0),
    ("This is not helpful", 0),
    ("I'm disappointed with the answer", 0),
    ("I'm not pleased with the answer", 0),
    ("This is not a satisfactory answer", 0),
    ("I'm not content with the answer", 0),
    ("I'm upset with the answer", 0),
    ("This is not what I was looking for", 0),
    ("I'm not okay with the answer", 0),
    ("This is not the answer I was expecting", 0),
    ("That's not the right answer", 0),
    ("This is not what I needed", 0),
    ("I'm dissatisfied with the answer", 0),
    ("This is not a good response", 0),
    ("I'm not happy with this solution", 0),
    ("I'm not satisfied with this solution", 0),
    ("This is not the solution I wanted", 0),
    ("I expected a better solution", 0),
    ("This is not a good solution", 0),
    ("This is not a helpful solution", 0),
    ("I'm disappointed with this solution", 0),
    ("I'm not pleased with this solution", 0),
    ("This is not a satisfactory solution", 0),
    ("I'm not content with this solution", 0),
    ("I'm upset with this solution", 0),
    ("This is not what I was expecting as a solution", 0),
    ("That's not the right solution", 0),
    ("This is not what I needed as a solution", 0),
    ("I'm dissatisfied with this solution", 0),
    ("This is not a good response as a solution", 0)])
    fine_tuning_data.extend([
    ("I'm happy with this", 1),
    ("Great", 1),
    ("Satisfied", 1),
    ("Pleased", 1),
    ("Wonderful", 1),
    ("Awesome", 1),
    ("Good job", 1),
    ("I'm content", 1),
    ("This is satisfactory", 1),
    ("I'm delighted", 1),
    ("I'm pleased with this", 1),
    ("This is what I wanted", 1),
    ("This is what I expected", 1),
    ("I'm happy with the answer", 1),
    ("I'm satisfied with the answer", 1),
    ("This is the answer I wanted", 1),
    ("This is a good answer", 1),
    ("This is helpful", 1),
    ("I'm pleased with the answer", 1),
    ("This is a satisfactory answer", 1),
    ("I'm content with the answer", 1),
    ("I'm happy with this solution", 1),
    ("I'm satisfied with this solution", 1),
    ("This is the solution I wanted", 1),
    ("This is a good solution", 1),
    ("This is a helpful solution", 1),
    ("I'm pleased with this solution", 1),
    ("This is a satisfactory solution", 1),
    ("I'm content with this solution", 1)])


    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    batch_encoding = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
    inputs = (batch_encoding["input_ids"], batch_encoding["attention_mask"])

    # Process fine-tuning data
    fine_tuning_texts = [item[0] for item in fine_tuning_data]
    fine_tuning_labels = tf.convert_to_tensor([item[1] for item in fine_tuning_data])

    tokenizer_ft = AutoTokenizer.from_pretrained(checkpoint)
    inputs_ft = tokenizer_ft(fine_tuning_texts, padding=True, truncation=True, return_tensors="tf")

    # Split the data into training and validation sets
    validation_split = 0.2  # 20% of the data for validation
    num_samples = len(fine_tuning_labels)
    num_validation_samples = int(validation_split * num_samples)

    train_inputs = (inputs_ft["input_ids"][:num_samples - num_validation_samples], inputs_ft["attention_mask"][:num_samples - num_validation_samples])
    train_labels = fine_tuning_labels[:-num_validation_samples]

    val_inputs = (inputs_ft["input_ids"][-num_validation_samples:], inputs_ft["attention_mask"][-num_validation_samples:])
    val_labels = fine_tuning_labels[-num_validation_samples:]

    # Convert input_ids and attention_mask to tensors
    train_input_ids_tensor = tf.convert_to_tensor(train_inputs[0])
    train_attention_mask_tensor = tf.convert_to_tensor(train_inputs[1])
    val_input_ids_tensor = tf.convert_to_tensor(val_inputs[0])
    val_attention_mask_tensor = tf.convert_to_tensor(val_inputs[1])

    # Fine-tune the model with additional data
    model_ft = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_ft.compile(optimizer=optimizer, loss=loss)

    # Then use these tensors for model training
    history = model_ft.fit(
        (train_input_ids_tensor, train_attention_mask_tensor),
        train_labels,
        epochs=5,
        validation_data=((val_input_ids_tensor, val_attention_mask_tensor), val_labels)
    )

    # Print training and validation losses for each epoch
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Make predictions with the fine-tuned model
    outputs = model_ft(inputs)
    predictions = tf.math.softmax(outputs.logits, axis=-1)
    sentiments = ['neg' if max(i) == i[0] else 'pos' for i in predictions]
    sentiments_correct = ['neg', 'pos', 'pos', 'neg', 'neg', 'pos', 'neg', 'pos', 'neg']

    # Calculate accuracy
    correct_predictions = sum(1 for pred, correct in zip(sentiments, sentiments_correct) if pred == correct)
    total_predictions = len(sentiments)
    accuracy = correct_predictions / total_predictions
    print("Test Accuracy:", accuracy)
    return model_ft

def model_use(raw_input, model):
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    batch_encoding = tokenizer(raw_input, padding=True, truncation=True, return_tensors="tf")
    inputs = (batch_encoding["input_ids"], batch_encoding["attention_mask"])
    outputs = model(inputs)
    predictions = tf.math.softmax(outputs.logits, axis=-1)
    #print(['neg' if max(i) == i[0] else 'pos' for i in predictions])
     # Get the index of the maximum value in predictions[0]
    max_index = tf.argmax(predictions[0], axis=-1).numpy()
    # Assign sentiment based on the max_index
    sentiment = 'neg' if max_index == 0 else 'pos'
    return sentiment

def sentiment():    
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
    return model

if __name__ == "__main__":  
    model = sentiment()
    print("Demonstration of working of the Sentiment Analysis")
    print("This is not what I was looking for: " , model_use("This is not what I was looking for", model))
    print("Again: ",model_use("Again", model))
    print("This is perfect: ", model_use("This is perfect", model))
    print("That was incorrect: ", model_use("That was incorrect", model))
    print("I am satisfied:", model_use("I am satisfied", model))
 

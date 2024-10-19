from sklearn.metrics import accuracy_score, classification_report
from constants import trained_models_folder_name, nb_intent_classifier_model_file_name, nb_vectorizer_filename, \
    encoders_folder_name, svm_encoder_classes_filename, svm_intent_classifier_model_file_name
import pickle
from load_data import load_test_data
from util import svm_encode_sentences


def load_validation_dictionary():
    val_dic = load_test_data()
    return val_dic


def load_validation_sentences(val_dic):
    val_sentences = []

    for i in val_dic:
        val_sentences.append(i['text'])  # Append the text
    return val_sentences


def load_validation_labels(val_dic):
    val_labels = []

    for i in val_dic:
        val_labels.append(i['label'])  # Append the text
    return val_labels


def validate_runtime_model(trained_model, X_val, val_labels):
    predictions = trained_model.predict(X_val)

    accuracy = accuracy_score(val_labels, predictions)

    print(f"Validation accuracy: {accuracy}")

    print(classification_report(val_labels, predictions))


# Method to load the saved encoder
def svm_load_encoder():
    path = encoders_folder_name + "/" + svm_encoder_classes_filename
    with open(path, 'rb') as f:
        encoder = pickle.load(f)
    return encoder


# Method to load the saved model
def svm_load_model():
    path = trained_models_folder_name + "/" + svm_intent_classifier_model_file_name
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


# Run validation for the trained model
def svm_validate_model(trained_model, sentences, labels):
    print("Validating SVM Model...")
    # Predict for the sentences
    predicted_labels = trained_model.predict(sentences)

    # Variable for counting correct predictions
    correct_count = 0

    # Iterate through the actual labels
    for item in range(len(labels)):
        if predicted_labels[item] == labels[item]:
            # If the prediction is correct, increment the correct count
            correct_count += 1

    print("Predicted {} correctly out of {}".format(correct_count, len(labels)))
    print("SVM Model accuracy: {}%".format(round(correct_count / len(labels) * 100), 2))
    print("\n")


# Validates the pickle file for SVM model
def svm_validate_saved_model():
    model = svm_load_model()  # Load the model

    # Load the saved encoder
    label_encoder = svm_load_encoder()
    # Load the validation dictionary
    validation_dictionary = load_validation_dictionary()

    # Load validation sentences and labels
    val_sentences = load_validation_sentences(validation_dictionary)
    val_labels = load_validation_labels(validation_dictionary)

    # Encode validation sentences
    val_X = svm_encode_sentences(val_sentences)

    # Encode validation labels
    encoded_validation_labels = label_encoder.fit_transform(val_labels)

    # Run model validation
    svm_validate_model(model, val_X, encoded_validation_labels)

    # Test a sample sentence as well
    # Encoding the sample sentence
    sample_sentence = "Give The Foundation by Isaac Asmiov 5 stars"
    sample_encoded_sentence = svm_encode_sentences([sample_sentence])

    # Infer with the model
    pred = model.predict(sample_encoded_sentence)
    # Get the predicted label
    predicted_label = label_encoder.inverse_transform(pred)

    print(f"Predicted label for sample sentence, \"{sample_sentence}\" : " + str(predicted_label[0]))


def nb_load_saved_model():
    path = trained_models_folder_name + "/" + nb_intent_classifier_model_file_name
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def nb_load_saved_vectorizer():
    path = encoders_folder_name + "/" + nb_vectorizer_filename
    with open(path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


def nb_validate_saved_model():
    print("Loading the saved model and the vectorizer")
    model = nb_load_saved_model()
    vectorizer = nb_load_saved_vectorizer()
    print("Loaded the saved model and the vectorizer")

    validation_dictionary = load_validation_dictionary()

    val_sentences = load_validation_sentences(validation_dictionary)
    val_labels = load_validation_labels(validation_dictionary)

    # Vectorize the validation sentences
    X_val = vectorizer.transform(val_sentences)

    # print("Validating saved model")
    validate_runtime_model(model, X_val, val_labels)

    sample_sentence = "Order a table for 5 at Sun Food"
    sample_vec_sentence = vectorizer.transform([sample_sentence])

    prediction = model.predict(sample_vec_sentence)
    print(f"Prediction for sample sentece, \"{sample_sentence}\" is: {prediction[0]}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from load_data import load_training_data, load_test_data
from model_validator import nb_validate_saved_model, svm_validate_model, \
    validate_runtime_model, svm_validate_saved_model
from util import svm_encode_sentences, svm_save_encoder_classes, svm_save_model_pickle, nb_save_model_pickle, \
    nb_save_vectorizer


# Returns the training sentences from a given training dictionary
def get_training_sentences(training_dic):
    sen_training = []
    # Iterate through training dictionary
    for i in training_dic:
        sen_training.append(i['text'])  # Append the text
    return sen_training


# Returns the training labels from a given training dictionary
def get_training_labels(training_dic):
    labels_training = []
    # Iterate through training dictionary
    for i in training_dic:
        labels_training.append(i['label'])  # Append the text
    return labels_training


# Returns the validation sentences from a given training dictionary
def get_validation_sentences(validation_dic):
    sen_validation = []
    for i in validation_dic:
        sen_validation.append(i['text'])
    return sen_validation


# Returns the validation labels from a given training dictionary
def get_validation_labels(validation_dic):
    labels_validation = []
    for i in validation_dic:
        labels_validation.append(i['label'])
    return labels_validation


# Trains the SVM Classifier for the problem
def train_svm_classifier():
    print('Initiating SVM Classifier\n')
    # Load training and validation data
    training_dic = load_training_data()
    val_dic = load_test_data()

    # Get training sentences and labels
    sen_training = get_training_sentences(training_dic)
    labels_training = get_training_labels(training_dic)

    # Get validation sentences and labels
    sen_val = get_validation_sentences(val_dic)
    labels_val = get_validation_labels(val_dic)

    print('Encoding sentences...\n')
    print('Encoding training sentences...')
    train_X = svm_encode_sentences(sen_training)  # Get encoded training sentences
    print('Encoded training sentences.\n')
    print('Encoding validation sentences...')
    val_X = svm_encode_sentences(sen_val)  # Get encoded validation sentences
    print('Encoded validation sentences.\n')

    print('Encoded sentences.\n')

    # Instantiate Label Encoder object
    label_encoder = LabelEncoder()

    # Encode training and validation labels
    labels_training = label_encoder.fit_transform(labels_training)
    labels_val = label_encoder.fit_transform(labels_val)

    print("First label after encoding: " + str(labels_training[0]))

    # Instantiate the model
    model = SVC(C=10)

    # Train the model
    print("Training the model...")
    model.fit(train_X, labels_training)
    print("Model Trained.\n")

    # Validate the trained model
    print("Validating the trained model...")
    svm_validate_model(model, val_X, labels_val)

    # Save the trained model and the fitted encoder classes
    svm_save_model_pickle(model)
    svm_save_encoder_classes(label_encoder)


# Trains the Naive Bayes Classifier for the problem
def train_nb_classifier():
    print("Initiating Naive Bayes classifier\n")
    # Load training and validation data
    training_dic = load_training_data()
    val_dic = load_test_data()

    # Get training sentences and labels
    sen_training = get_training_sentences(training_dic)
    labels_training = get_training_labels(training_dic)

    # Get validation sentences and labels
    sen_val = get_validation_sentences(val_dic)
    labels_val = get_validation_labels(val_dic)

    # Instantiate vectorizer
    vectorizer = TfidfVectorizer()

    # Instantiate the model
    model = MultinomialNB()

    # Vectorize training and validation sentences
    X_train = vectorizer.fit_transform(sen_training)
    X_val = vectorizer.transform(sen_val)

    print("Training the naive bayes model...")
    # Train the model
    model.fit(X_train, labels_training)
    print("Model training completed")

    # Validate the trained model
    print("Validating the trained model...")
    validate_runtime_model(model, X_val, labels_val)

    # Save the model and the vectorizer
    nb_save_model_pickle(model)
    nb_save_vectorizer(vectorizer)


if __name__ == "__main__":
    train_svm_classifier()
    svm_validate_saved_model()
    train_nb_classifier()
    nb_validate_saved_model()

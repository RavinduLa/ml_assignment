import spacy
import numpy as np
import pickle
from constants import trained_models_folder_name, nb_intent_classifier_model_file_name, nb_vectorizer_filename, \
    encoders_folder_name, svm_intent_classifier_model_file_name, svm_encoder_classes_filename


def process_nlp():
    nlp_loaded = spacy.load("en_core_web_md")
    # print("Number of vectors: {}".format(nlp_loaded.vocab.vectors_length))
    return nlp_loaded


def svm_encode_sentences(sentences):
    nlp = process_nlp()
    # Calculate the number of sentences
    n_sentences = len(sentences)
    X = np.zeros((n_sentences, 300))

    # Iterate over the sentences
    for idx, sentence in enumerate(sentences):
        # Pass each sentence to the nlp object to create a document
        doc = nlp(sentence)
        # Save the document's .vector attribute to the corresponding row in X
        X[idx, :] = doc.vector
    # print("Return type of X when encoding sentences" + type(X) )
    return X


def svm_save_model_pickle(trained_model):
    print("Saving the model...")
    path = trained_models_folder_name + "/" + svm_intent_classifier_model_file_name

    pickle.dump(trained_model, open(path, 'wb'))
    print("Saved the model")


# Saves the fitted encoder as a pickle based file
def svm_save_encoder_classes(fitted_encoder):
    print("Saving the encoder...")
    path = encoders_folder_name + "/" + svm_encoder_classes_filename
    pickle.dump(fitted_encoder, open(path, 'wb'))
    print("Saved the encoder")


# Saves the model as a pickle based file
def nb_save_model_pickle(trained_model):
    print("Saving the model...")
    path = trained_models_folder_name + "/" + nb_intent_classifier_model_file_name

    pickle.dump(trained_model, open(path, 'wb'))
    print("Saved the model")


def nb_save_vectorizer(vectorizer):
    print("Saving the vectorizer...")
    path = encoders_folder_name + "/" + nb_vectorizer_filename

    pickle.dump(vectorizer, open(path, 'wb'))
    print("Saved the vectorizer")

import json
import random


def load_training_data():
    training_data = []

    # load the data for each lass
    add_to_playlist_data = load_data_from_file(assemble_training_file_path("AddToPlaylist"), "AddToPlaylist")
    book_restaurant_data = load_data_from_file(assemble_training_file_path("BookRestaurant"), "BookRestaurant")
    get_weather_data = load_data_from_file(assemble_training_file_path("GetWeather"), "GetWeather")
    play_music_data = load_data_from_file(assemble_training_file_path("PlayMusic"), "PlayMusic")
    rate_book_data = load_data_from_file(assemble_training_file_path("RateBook"), "RateBook")
    search_creative_work_data = load_data_from_file(assemble_training_file_path("SearchCreativeWork"),
                                                    "SearchCreativeWork")
    search_screening_event_data = load_data_from_file(assemble_training_file_path("SearchScreeningEvent"),
                                                      "SearchScreeningEvent")
    # Iterate through each array and append items to training_data
    for i in add_to_playlist_data:
        training_data.append(i)
    for i in book_restaurant_data:
        training_data.append(i)
    for i in get_weather_data:
        training_data.append(i)
    for i in play_music_data:
        training_data.append(i)
    for i in rate_book_data:
        training_data.append(i)
    for i in search_creative_work_data:
        training_data.append(i)
    for i in search_screening_event_data:
        training_data.append(i)

    #print("Before shuffling training data,")
    #print("Training data array size: " + str(len(training_data)))
    #print("Training data sample: " + str(training_data[432]))

    #print("\n")

    # Shuffle the training data
    training_data = shuffle_data(training_data)

    #print("After shuffling training data,")
    #print("Training data array size: " + str(len(training_data)))
    #print("Test training data: " + str(training_data[432]))

    #print("\n")

    # Return the shuffled list
    return training_data


# Takes a data array and shuffles it and returns
def shuffle_data(data):
    random.shuffle(data)
    return data


def load_test_data():
    test_data = []
    add_to_playlist_data = load_data_from_file(assemble_test_file_path("AddToPlaylist"), "AddToPlaylist")
    book_restaurant_data = load_data_from_file(assemble_test_file_path("BookRestaurant"), "BookRestaurant")
    get_weather_data = load_data_from_file(assemble_test_file_path("GetWeather"), "GetWeather")
    play_music_data = load_data_from_file(assemble_test_file_path("PlayMusic"), "PlayMusic")
    rate_book_data = load_data_from_file(assemble_test_file_path("RateBook"), "RateBook")
    search_creative_work_data = load_data_from_file(assemble_test_file_path("SearchCreativeWork"), "SearchCreativeWork")
    search_screening_event_data = load_data_from_file(assemble_test_file_path("SearchScreeningEvent"), "SearchScreeningEvent")

    # Iterate through each array and append items to training_data
    for i in add_to_playlist_data:
        test_data.append(i)
    for i in book_restaurant_data:
        test_data.append(i)
    for i in get_weather_data:
        test_data.append(i)
    for i in play_music_data:
        test_data.append(i)
    for i in rate_book_data:
        test_data.append(i)
    for i in search_creative_work_data:
        test_data.append(i)
    for i in search_screening_event_data:
        test_data.append(i)

    #print("Before shuffling test data,")
    #print("Test data array size: " + str(len(test_data)))
    #print("Test data sample: " + str(test_data[432]))

    #print("\n")

    # Shuffle the training data
    test_data = shuffle_data(test_data)

    #print("After shuffling test data,")
    #print("Test data array size: " + str(len(test_data)))
    #print("Test training data: " + str(test_data[432]))

    #print("\n")

    # Return the shuffled list
    return test_data


# assembles the file path in the training directory for the given file_name
def assemble_training_file_path(file_name):
    rootPath = "dataset/train/"
    fileSuffix = ".json"
    file_path = rootPath + file_name + fileSuffix
    return file_path


# assembles the file path in the validate directory for the given file_name
def assemble_test_file_path(file_name):
    rootPath = "dataset/validate/"
    fileSuffix = ".json"
    file_path = rootPath + file_name + fileSuffix
    return file_path


# Load data and create a sentence array with dictionaries for a given file_path and a filename
def load_data_from_file(file_path, filename):
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    sentences = []
    for i in data[filename]:
        dictionaryItem = {'text': assemble_sentence(i['data']), 'label': filename}
        sentences.append(dictionaryItem)

    # print(sentences[1]['label'])
    # print(sentences[1]['text'])
    return sentences


# Assembles a sentence from a given array
def assemble_sentence(data):
    sentence = ""
    for i in data:
        sentence = sentence + i['text']
    # print('assembled sentence: ' + sentence)
    return sentence

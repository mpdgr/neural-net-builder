import logging as log
import re
import tkinter as tk
import jsonpickle


def load_data(path):
    with open(path, 'r') as file:
        encoded_data = file.read()
    data = jsonpickle.decode(encoded_data)
    print(type(data))
    return data


# run prediction on input sentence
def predict_on_input(inp):
    # lowercase and substitute anything but letters and spaces with spaces, split to tokens
    inp = re.sub(r'[^a-z ]', ' ', inp.lower())
    review_words = inp.split(" ")

    # create empty input vector
    review_vector = [0] * len(dictionary)

    # create vector for given input encoding each word/token into correct index basing on dictionary
    for word in review_words:
        if word.strip() in dictionary:
            review_vector[dictionary[word.strip()]] = 1

    # compute prediction
    prediction = network.predict(review_vector)
    log.info(f'prediction value: {prediction}')

    if prediction[0] >= 0.50:
        return 1
    elif prediction[0] < 0.50:
        return 0
    else:
        raise Exception(f'Prediction error! prediction: {prediction}')


# format info label
def format_positive():
    score_field.config(text='POSITIVE', fg="green")


def format_negative():
    score_field.config(text='NEGATIVE', fg='red')


log.getLogger().setLevel(log.INFO)

# load network and dictionary
path_network = 'pre_trained/network.json'
path_dictionary = 'pre_trained/dictionary.json'
network = load_data(path_network)
dictionary = load_data(path_dictionary)
log.info(f'Network and dictionary loaded')
log.info(f'Dictionary size: {len(dictionary)}')


# create widget to test the network on user input

# input change listener
def on_input(event):
    inp = entry_field.get("1.0", tk.END)

    if len(inp) < 2:
        score_field.config(text="")
        log.info(f'input too short')
        return

    prediction = predict_on_input(inp)
    if prediction == 1:
        format_positive()
        log.info(f'prediction: positive')
    else:
        format_negative()
        log.info(f'prediction: negative')


# widget
root = tk.Tk()
root.title("Review evaluator")
root.config(padx=20)

header = tk.Label(root, text="Input movie review...", font=('Consolas', 12), padx=20, pady=20)
header.pack()

entry_field = tk.Text(root, font=('Consolas', 12), bg='lightgrey', fg='black', bd=5, relief='flat',
                      width=50, height=5, wrap="word")
entry_field.pack()
entry_field.bind("<KeyRelease>", on_input)

score_label = tk.Label(root, text="Score:", font=('Consolas', 12), pady=20)
score_field = tk.Label(root, text="", font=('Consolas', 12, 'bold'))
score_label.pack()
score_field.pack()
frame = tk.Frame(root)
frame.pack(pady=(0, 30))
entry_field.focus_set()
root.mainloop()

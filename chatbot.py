from tensorflow.keras.models import load_model
from utils import *

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('./intents.json').read())

words = pickle.load(open('./pickles/words.pkl', 'rb'))
classes = pickle.load(open('./pickles/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.h5')


def clean_sentence(sentence: str):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        str.lower(word)) for word in sentence_words]

    return sentence_words


def bag_of_words(sentence: str):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence: str):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = list([])
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

os.system("clear")
print('Go! Nao is currently running!')

while True:
    message = input('')
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

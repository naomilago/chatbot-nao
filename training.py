from utils import *

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('./intents.json').read())

words, classes, documents = list([]), list([]), list([])
punctuation = list(['?', ',', '.', '-', '[', ']', '"', '\'', "!"])

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word)
         for word in words if word not in punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('pickles/words.pkl', 'wb'))
pickle.dump(classes, open('pickles/classes.pkl', 'wb'))

training = list([])
empty_output = [0] * len(classes)

for document in documents:
    bag = list([])
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        str.lower(word)) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(empty_output)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

x_train = list(training[:, 0])
y_train = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(x_train), np.array(y_train),
                 epochs=200, batch_size=5, verbose=1)
model.save('model/chatbot_model.h5', hist)

print('The model was trained successfully and stored at ./model/chatbot_model.model')

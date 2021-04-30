from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.core.files.storage import FileSystemStorage
import os, json
from django.urls import reverse

import pickle
import json
import tflearn
import tensorflow as tf
import numpy as np
import random
import nltk

import urllib.request
import urllib.parse
import re
from bs4 import BeautifulSoup


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# restoring all the data structures
data = pickle.load( open( "MYSITE/data/training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('MYSITE/data/intents.json') as json_data:
    intents = json.load(json_data)


# Building neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Defining model and setting up tensorboard
model = tflearn.DNN(net)

model.load('MYSITE/data/model.tflearn')

# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    
    # tokenizing the pattern
    sentence_words = sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    
    # generating bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

ERROR_THRESHOLD = 0.30
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def responses(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return [random.choice(i['responses']), i['tag']]

            results.pop(0)
    else:
        return 0
        
def index(request):
    return render(request, 'template.html')

def result(request, text):
    if "video" in text or "play" in text:
        query_string = urllib.parse.urlencode({"search_query" : text})
        html_content = urllib.request.urlopen("https://www.youtube.com/results?" + query_string)
        search_results = re.findall(r'url\":\"\/watch\?v=(.{11})', html_content.read().decode())
        if "play" in text:
            search_results = search_results[0:5]
        else:
            search_results = search_results[0:10]
        response = {
        'text': "https://www.youtube.com/embed/" + random.choice(search_results),
        'intent': "10"
        }
        return HttpResponse(json.dumps(response), content_type="application/json")
    else:       
        r = responses(text)
        if r != 0:
            if "meme" in text or "joke" in text:
                if r[0] == "joke":
                    html = urllib.request.urlopen('https://imgur.com/search?q=joke')
                elif r[0] == "meme": 
                    html = urllib.request.urlopen('https://imgur.com/search?q=meme')
                bs = BeautifulSoup(html, 'html.parser')
                images = bs.find_all('img', {'src':re.compile('.jpg')})
                response = {
                    'text': random.choice(images)['src'],
                    'intent': "11.5"
                }
                return HttpResponse(json.dumps(response), content_type="application/json")
            else:
                response = {
                    'text': r[0],
                    'intent': r[1]
                }
                return HttpResponse(json.dumps(response), content_type="application/json")
        else:
            response = {
                    'text': "Sorry, didnt get that!!",
                    'intent': "-1"
                }
            return HttpResponse(json.dumps(response), content_type="application/json")

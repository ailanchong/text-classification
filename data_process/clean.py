# -*- coding:utf-8 -*- 
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import re
def get_set(filepath):
    result = set()
    with open(filepath, 'r') as file_in:
        for line in file_in:
            result.add(line.strip())
    return result
def get_map(filepath):
    result = {}
    with open(filepath,'r') as file_in:
        for line in file_in:
            line = line.strip().split(",")
            print(line)
            result[line[0]] = line[1]
    return result

def init_datastructure(swearfile, corrfile, knowsetfile, keysetfile, commonsetfile):
    swear_words = get_set(swearfile)
    corr_map = get_map(corrfile)
    knowset = get_set(knowsetfile)
    keysetfile = get_set(keysetfile)
    commonsetfile = get_set(commonsetfile)
    return swear_words, corr_map, knowset, keysetfile, commonsetfile

def delete_at(words):
    result = []
    for word in words:
        if word[0] == '@':
            result.append(word[1:])
        else:
            result.append(word)
    return result


def delete_http(words):
    result = []
    for word in words:
        if word[0:4] == 'http':
            pass
        else:
            result.append(word)
    return result

def combine_swear_words(text, swear_words):
    text_len = len(text)
    result = []
    temp = ""
    i = 0
    while i < text_len:
        if((len(text[i]) != 1) or (not text[i].isalpha())):
            if(temp in swear_words):
                result.append(temp)
                temp = ""
            else:
                temp = ""
            if(text[i].isalpha or text[i] in string.punctuation):
                result.append(text[i])
        else:
            temp = temp + text[i]
            if (temp in swear_words):
                result.append(temp)
                temp = ""
        i += 1

    if(temp in swear_words):
        result.append(temp)
    return result

def split_words(text, corr_map):
    result = []
    for word in text:
        if(corr_map.get(word) != None):
            temp = corr_map.get(word)
            temp = temp.split(" ")
            result.extend(temp)
        else:
            result.append(word)
    return result

def get_keyword(text, knowset, keyset, commonset):
    result = []
    for word in text:
        if word in knowset:
            result.append(word)
            continue
        mark = False
        for keyword in keyset:
            if keyword in word:
                if(keyword == "kill" and "skill" in word):
                    continue
                else:
                    result.append(keyword)
                    mark = True
        if(not mark):
            if(word in commonset):
                result.append(word)
    return result
def clean(comment, swear_words, corr_map, knowset, keyset, commonset):
    APPO = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "I would",
    "i'd" : "I had",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will",
    "didn't": "did not",
    "tryin'":"trying"
    }
    comment = comment.lower()
    comment = re.sub("\\n"," ",comment)
    comment = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","ip",comment) #去掉IP
    comment = re.sub("\[\[.*\]","username",comment) #去掉用户名
    tokenizer=TweetTokenizer()
    lem = WordNetLemmatizer()
    words=tokenizer.tokenize(comment)
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words=[lem.lemmatize(word, "n") for word in words]
    words = delete_at(words) #去掉@
    words = delete_http(words) #去掉http
    words = combine_swear_words(words, swear_words) #合并f_u_c_k >>> fuck
    words = split_words(words, corr_map) #拆分 mothafuckin >> mother fuck
    #words = get_keyword(words, knowset, keyset, commonset) #提取脏话主干，并且保留特定语料的常用词
    clean_sent=" ".join(words)
    return clean_sent











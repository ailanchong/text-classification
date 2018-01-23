from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
import numpy as np 


train = ['bird is not . ? a ', 'cat \
 wdad', "dog isn't adas  qqw", 'fish asd', '__na__']
test = ['birwd', 'cat', 'dog', 'fish']
corpus = train + test
cv = CountVectorizer()
cv.fit(corpus)
array = cv.transform(train)
print(cv.get_feature_names())
print(array.toarray())

print(cv.vocabulary_)
'''
test = ['bird', 'cat', 'dog', 'fish']
cv.fit(test)
array = cv.transform(train)
print(cv.get_feature_names())
print(array.toarray())
'''


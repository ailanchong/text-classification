from pyfasttext import FastText
'''
model = FastText()
model.skipgram(input="./clean_corpus", output='model', epoch=100, lr=0.7)
print(model.nwords)
'''
#model = fasttext.load_model('../data/model.bin')
model = FastText('../data/model.bin')
def get_set(filepath):
    result = set()
    with open(filepath, 'r') as file_in:
        for line in file_in:
            result.add(line.strip())
    return result

vocab_in = get_set("../data/vocab_in")
vocab_out = get_set("../data/vocab_out")
in2out_file = open("../data/in2out_map",'w')
for word in vocab_out:
    candi_list = model.nearest_neighbors(word, k=10)
    mark = False
    for temp in candi_list:
        if(temp[0] in vocab_in):
            in2out_file.write("\t".join([word, temp[0]]) + "\n")
            mark = True
            break
    if(not mark):
        in2out_file.write("\t".join([word, "none"]) + "\n")
    
in2out_file.close()        


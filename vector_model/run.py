import gensim.models as gsm
import sklearn as sk
import pickle 

# path to pretrained emoji's and word vectors.
PATH_emoji2vec = "/home/susuresh/emoji2vec/pre-trained/emoji2vec.bin"
PATH_word2vec = "/home/susuresh/Downloads/GoogleNews-vectors-negative300.bin.gz"

# load all emoji vectors from emoji2vec
e2v = gsm.KeyedVectors.load_word2vec_format(PATH_emoji2vec, binary=True)

# load all word vectors from word2vec
w2v = gsm.KeyedVectors.load_word2vec_format(PATH_word2vec, binary=True)

print("Loading vectors done")
# load sentence

def getEmoji(sentence):
	sentence_words = sentence.split(" ")
	wvecs = []
	for word in sentence_words:
		if word in w2v.vocab:
			vector = w2v[word]
			wvecs.append(vector)

	pred_emoji_list = []
	for wvec in wvecs:
		pred_emoji_list.append(e2v.similar_by_vector(wvec,topn=2))

	return pred_emoji_list

sentences = ["9 perfect last minute summer trip ideas",
			"Shop shoes and clothes at PUMA with your reward points & get 50% value back",
			"Save up to 15% on dining with Citi cards at these new premium restaurants"]
pel = []
for sentence in sentences:
	print(sentence)
	pel.append(getEmoji(sentence))

with open("pred_emojis.txt", "w",encoding='utf-8') as output:
    output.write(str(pel))
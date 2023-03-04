from gensim.models import Word2Vec
import json
import pickle
import tqdm

# sentence = [["above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between",
#              "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from",
#              "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on",
#              "near", "of", "on", "on back of",
#              "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on",
#              "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]]
#
# model = Word2Vec(sentences=sentence, vector_size=5, window=5, min_count=1, workers=4)
# print(model.wv['on'])
# print(model.wv['riding'])
# print(model.wv['of'])
# print(model.wv['growing on'])
# print(model.wv['sitting on'])
# print(model.wv['near'])

# vec = model.wv['on']
# print(vec)

# from gensim.models import Word2Vec
# from gensim.models.word2vec import LineSentence
# import numpy as np
#
# # sentence = LineSentence("content.txt")  #如果语料是文件，可以使用LineSentence准备训练语料
# sentence = [["小明", "今天", "要", "去", '少年宫', "游泳"]]  # 准备训练预料
# model = Word2Vec(sentences = sentence, vector_size=5, window=5, min_count=1, workers=4) # 生成模型
# word_vectors = model.wv['小明'] # 输出词语的向量映射
# print(word_vectors) # [-0.06810732 -0.01892805  0.11537147 -0.15043278 -0.0787221 ]
# moresentence = [["小明", "和", "小明", "哥哥", "不要", "去", '少年宫', "游泳"]] # 准备训练预料
# model.train(corpus_iterable = moresentence, epochs = 1, total_words = 1) # 训练模型
# model.save('train_demo.model') # 保存模型
# model = Word2Vec.load('train_demo.model')  # 加载模型
# # 使用模型
# result = model.wv.most_similar(positive=['今天', '游泳'], negative=['少年宫'], topn=2) # 使用模型找出相近的10个词，'今天', '游泳'对相似性有正面贡献，'少年宫'有负面贡献
# print(result) # [('去', 0.714894711971283), ('要', -0.5734316110610962)]
# distance = model.wv.distance("少年宫", "小明") # 两个单词的距离
# print(distance) # 0.22581267356872559
path = "../em_E.pk"

vocab = json.load(open("../VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k): v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v) for k, v in vocab["predicate_to_idx"].items()}

l = pickle.load(open(path, "rb"))

sentence = []
index = 0

for data in l:
    labels = data["labels"]
    relation_tuple = data["relations"]
    sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
    sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
    for i in range(len(relation_tuple)):
        sentence.append(list())
        sentence[index].append(idx2lb[sub_lbs[i]])
        sentence[index].append(idx2pred[rels[i]])
        sentence[index].append(idx2lb[obj_lbs[i]])
        index = index + 1

model = Word2Vec(sentences=sentence, window=2, min_count=1, workers=4)

predfeatures = []
for item in idx2pred:
    predfeatures.append(model.wv[idx2pred[item]])
pass

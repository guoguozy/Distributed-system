import nltk
import jieba
import collections
def get_target_sen(file_path):
    """
    读取目标文件
    """
    target_sen = []
    with open(file_path,'r',encoding='utf-8') as f:
        context = f.readlines()
        for sen in context:
            sen = sen.strip()
            if len(sen) == 0:
                continue
            temp = nltk.word_tokenize(sen.lower())  #用nltk进行分词，并把所有的单词转换成小写
            temp.insert(0,'<bos>')                  #在句子开头插入<bos>
            temp.append('<eos>')                    #在句子结尾插入<eos>
            target_sen.append(temp)
    return target_sen

def get_score_sen(file_path):
    """
    读取源文件
    """
    score_sen = []
    with open(file_path,'r',encoding='utf-8') as f:
        context = f.readlines()
        for sen in context:
            sen = sen.strip()
            word_list = jieba.cut(sen,cut_all=False)   #用jieba分词
            sen = ' '.join(word_list)
            temp_list = sen.split()
            temp_list.insert(0,'<bos>')                #在句子开头插入<bos>
            temp_list.append('<eos>')                  #在句子结尾插入<eos>
            score_sen.append(temp_list)
    return score_sen

def word2id(word_to_id,sen_list):
    """
    把单词转换成对应的数字
    """
    ID_sen = []
    for sen in sen_list:
        id_sen = []
        for word in sen:
            vocab_id = word_to_id[word]
            id_sen += [vocab_id]
        ID_sen += [id_sen]
    return ID_sen

def build_vocab(data):
    """
    建立单词到数字的映射
    """
    counter = collections.Counter(data)               # 用 Counter 统计单词出现的次数，为了之后按单词出现次数的多少来排序
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))  # 单词到整数的映射
    return word_to_id

train_source = get_score_sen('.\\dataset_10000\\train_source_8000.txt')
train_target = get_target_sen('.\\dataset_10000\\train_target_8000.txt')
test_source = get_score_sen('.\\dataset_10000\\test_source_1000.txt')
test_target = get_score_sen('.\\dataset_10000\\test_target_1000.txt')

sequence_lengths = [len(seq) for seq in train_target]

source_vocab = []             #存储源语言句子的所有词语
for sen in train_source:
    for word in sen:
        source_vocab.append(word)

for sen in test_source:
    for word in sen:
        source_vocab.append(word)



source_word_to_id = build_vocab(source_vocab)
source_word_to_id['<eos>'] = 0
source_word_to_id['的'] = 3
train_source_id = word2id(source_word_to_id,train_source)
test_source_id = word2id(source_word_to_id,test_source)


target_vocab = []       #存储目标语言句子的所有词语
for sen in train_target:
    for word in sen:
        target_vocab.append(word)

for sen in test_target:
    for word in sen:
        target_vocab.append(word)

# for sen in dev_target:
#     for word in sen:
#         target_vocab.append(word)

target_word_to_id = build_vocab(target_vocab)
target_word_to_id['<eos>'] = 0
target_word_to_id['the'] = 3
target_word_to_id[','] = 1
#print(target_word_to_id)
train_target_id = word2id(target_word_to_id, train_target)
test_target_id = word2id(target_word_to_id, test_target)
# dev_target_id = word2id(target_word_to_id, dev_target)


target_vocab = list(set(target_vocab))
target_id2word = {v:k for k, v in target_word_to_id.items()}















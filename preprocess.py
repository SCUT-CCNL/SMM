import numpy as np
import re
import json
import jieba
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class Data():
    def __init__(self, max_len=0):
        self.s1s, self.s2s, self.labels, self.features = [], [], [], []
        self.index, self.max_len = 0, max_len
        self.src_s1s, self.src_s2s, self.src_labels, self.src_features = [], [], [], []
        self.src_index = 0

    def open_file(self):
        pass

    def is_available(self):
        if self.index < self.data_size:
            return True
        else:
            return False

    def reset_index(self):
        self.index = 0

    def next(self):
        if (self.is_available()):
            self.index += 1
            return self.data[self.index - 1]
        else:
            return

    def gen_data(self, word2idx, max_len):
        s1_mats, s2_mats = [], []
        s1_seq_len, s2_seq_len = [], []

        for i in range(self.data_size):
            s1 = self.s1s[i]
            s2 = self.s2s[i]
            s1_seq_len.append(len(s1))
            s2_seq_len.append(len(s2))

            # [1, d0, s]
            s1_idx = []
            for w in s1:
                if w in word2idx:
                    s1_idx.append(word2idx[w])
                else:
                    print('word not in vocab')
            for _ in range(max_len-len(s1_idx)):
                s1_idx.append(0)
            s2_idx = []
            for w in s2:
                if w in word2idx:
                    s2_idx.append(word2idx[w])
                else:
                    print('word not in vocab')
            for _ in range(max_len-len(s2_idx)):
                s2_idx.append(0)
            s1_mats.append(s1_idx)
            s2_mats.append(s2_idx)

        # [batch_size, d0, s]
        total_s1s = np.asarray(s1_mats)
        total_s2s = np.asarray(s2_mats)

        return total_s1s, total_s2s#, s1_seq_len, s2_seq_len

    '''
    def next_batch(self, total_s1s, total_s2s, batch_size):
        batch_size = min(self.data_size - self.index, batch_size)

        # [batch_size, d0, s]
        batch_s1s = total_s1s[self.index:self.index + batch_size]
        batch_s2s = total_s2s[self.index:self.index + batch_size]
        batch_labels = self.labels[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size] #not useful anymore

        self.index += batch_size

        return batch_s1s, batch_s2s, batch_labels, batch_features#, s1_seq_len, s2_seq_len
    '''

    def next_batch(self, total_s1s, total_s2s, labels, batch_size):
        batch_size = min(self.data_size - self.index, batch_size)

        # [batch_size, d0, s]
        batch_s1s = total_s1s[self.index:self.index + batch_size]
        batch_s2s = total_s2s[self.index:self.index + batch_size]
        batch_labels = labels[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        return batch_s1s, batch_s2s, batch_labels, batch_features#, s1_seq_len, s2_seq_len
class PI(Data):
    def load_data(self,src_data):
        len_thres = 100
        for i in range(len(src_data)):
            s1 = clean_str(src_data.iloc[i].question1).strip().split(' ')
            s2 = clean_str(src_data.iloc[i].question2).strip().split(' ')
            label = src_data.iloc[i].is_duplicate
            #limit sentences to len_thres
            if len(s1) >= len_thres:
                s1 = s1[:len_thres + 1]
            if len(s2) >= len_thres:
                s2 = s2[:len_thres + 1]
            self.s1s.append(s1)
            self.s2s.append(s2)
            self.labels.append(label)
            self.features.append([len(s1), len(s2)])

            local_max_len = max(len(s1), len(s2))
            if local_max_len > self.max_len:
                self.max_len = local_max_len
        self.data_size = len(self.s1s)
        self.num_features = len(self.features[0])
import re

def get_word_list(s1):
    regEx = re.compile('[\\W]*')    
    res = re.compile(r"([\u4e00-\u9fa5])")   

    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)

    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  

    return  list_word1
class Chinese_PI(Data):
    def load_data(self,src_data):
        len_thres = 100
        #thu = thulac.thulac(seg_only=True)
        for i in range(len(src_data)):
            s1 = list(jieba.cut(src_data.iloc[i].question1.lower()))
            s2 = list(jieba.cut(src_data.iloc[i].question2.lower()))
            #s1 = get_word_list(src_data.iloc[i].question1)
            #s2 = get_word_list(src_data.iloc[i].question2)
            label = src_data.iloc[i].is_duplicate
            #limit sentences to len_thres
            if len(s1) >= len_thres:
                s1 = s1[:len_thres + 1]
            if len(s2) >= len_thres:
                s2 = s2[:len_thres + 1]
            self.s1s.append(s1)
            self.s2s.append(s2)
            self.labels.append(label)
            self.features.append([len(s1), len(s2)])

            local_max_len = max(len(s1), len(s2))
            if local_max_len > self.max_len:
                self.max_len = local_max_len
        self.data_size = len(self.s1s)
        self.num_features = len(self.features[0])

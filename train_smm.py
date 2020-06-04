import tensorflow as tf
import numpy as np
import sys
import codecs
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument('-s',type=str,help='source data path')
parser.add_argument('-t',type=str,help='target data path')
parser.add_argument('-c',type=str,default='1',help='chinese w2v')
args = parser.parse_args()
np.set_printoptions(threshold=np.inf)

from preprocess import PI,Chinese_PI
from Semantic_Matching import SMM
from sklearn import linear_model, svm
from sklearn.externals import joblib
from sklearn import metrics
import pandas as pd

RandSeed = 1234
def test_accuracy(pred, true):
    if len(pred) != len(true):
        print("error: the length of two lists are not the same")
        return 0
    else:
        count = 0
        for i in range(len(pred)):
            if pred[i] == true[i]:
                count += 1
        return float(count)/len(pred)

def F_score1(pred, true):
    if len(pred) != len(true):
        print("error: the length of two lists are not the same")
        return 0
    else:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(pred)):
            if pred[i] == true[i] and pred[i] == 1:
                TP += 1
            elif pred[i] == true[i] and pred[i] == 0:
                TN += 1
            elif pred[i] != true[i] and pred[i] == 1:
                FP += 1
            elif pred[i] != true[i] and pred[i] == 0:
                FN += 1
        if (TP+FP) == 0:
            precision = 0
        else:
            precision = float(TP)/(TP+FP)
        if (TP+FN) == 0:
            precision = 0
        else:
            recall = float(TP)/(TP+FN)
        if precision ==0 or recall ==0:
            F_score =0
        else:
            F_score = 2*precision*recall/(precision+recall)
        return F_score

def re_norm(A):
    B = A.copy()
    for i in range(A.shape[0]):
        B[i][i] = 1.0
        for j in range(i+1, A.shape[0]):
            B[i][j] = B[i][j]/np.sqrt(np.abs(B[i][i]*B[j][j]))
            B[j][i] = B[i][j]
    return B

def norm_v(x, mu, sigma):
    #score = 1/(sigma*def_v) * np.exp(-(x-mu)**2 / (2 * sigma**2))
    score = np.exp(-(x-mu)**2 / (2 * sigma**2))
    return score

def print_matrix(A, meta):
    print (meta,)
    assert A.ndim == 2
    for i in range(len(A)):
        outstr = ""
        for v in A[i,:]:
            outstr += "%.3f "%v
        print ("  " + outstr.strip())
    print

def gen_vocab(data_list):
    word_to_idx = {}
    idx = 1
    for data_type in data_list:
        for i in range(data_type.data_size):
            s1 = data_type.s1s[i]
            for word in s1:
                if word not in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1
            s2 = data_type.s2s[i]
            for word in s2:
                if word not in word_to_idx:
                    word_to_idx[word] = idx
                    idx += 1
    return word_to_idx


def gen_trained_word_embedding(word2id):
    embeddings_index = {}
    f = open('../data/glove.840B.300d.txt', 'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    np.random.seed(RandSeed)
    embedding_matrix = np.random.uniform(-0.01, 0.01, (len(word2id)+1, 300))
    embedding_matrix[0] = 0
    # embedding_matrix = np.zeros((len(self.word2id), self.word_embed_size))
    vocab_size = len(word2id)
    pretrained_size = 0
    for word, i in word2id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            pretrained_size += 1
            embedding_matrix[i] = embedding_vector

    print('vocab size:%d\t pretrained size:%d' % (vocab_size, pretrained_size))

    return embedding_matrix
def gen_trained_chi_embedding(word2id):
    embeddings_index = {}
    f = codecs.open('../data/chi_w2v_skigram.txt','r',encoding='utf-8')
    first_line=True
    for line in f:
        if first_line:
            first_line = False
            continue
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    np.random.seed(RandSeed)
    embedding_matrix = np.random.uniform(-0.01, 0.01, (len(word2id)+1, 300))
    embedding_matrix[0] = 0
    # embedding_matrix = np.zeros((len(self.word2id), self.word_embed_size))
    vocab_size = len(word2id)
    pretrained_size = 0
    for word, i in word2id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            
            # words not found in embedding index will be all-zeros.
            pretrained_size += 1
            embedding_matrix[i] = embedding_vector

    print('vocab size:%d\t pretrained size:%d' % (vocab_size, pretrained_size))

    return embedding_matrix

def train_test_split(data,split_point=0.8):
    train_size = int(len(data)*0.8)
    return data[:train_size],data[train_size:]

def train(src,tgtrain,tgtest,lr, w, l2_reg, epoch, batch_size, model_type, num_layers, data_type, num_classes=2):
    src_train_data = PI()
    tgt_train_data = PI()
    test_data = PI()
    if args.c=='1':
        src_train_data = Chinese_PI()
        tgt_train_data = Chinese_PI()
        test_data = Chinese_PI()
    src_train_data.load_data(src)
    tgt_train_data.load_data(tgtrain)
    test_data.load_data(tgtest)
    data_list = [src_train_data, tgt_train_data, test_data]
    word2idx = gen_vocab(data_list)
    embedding_weight = None
    if args.c=='1':
        embedding_weight = gen_trained_chi_embedding(word2idx)
    else:
        embedding_weight = gen_trained_word_embedding(word2idx)

    max_len = max(src_train_data.max_len, tgt_train_data.max_len)

    src_total_s1, src_total_s2 = src_train_data.gen_data(word2idx=word2idx, max_len = max_len)
    tgt_total_s1, tgt_total_s2 = tgt_train_data.gen_data(word2idx=word2idx, max_len = max_len)
    test_total_s1, test_total_s2 = test_data.gen_data(word2idx=word2idx, max_len=max_len)

    print("=" * 50)
    print("src training data size:", src_train_data.data_size)
    print("training max len:", max_len)
    print("=" * 50)

    model = SMM(s=max_len, w=w, l2_reg=l2_reg, model_type=model_type, embeddings=embedding_weight, num_features=src_train_data.num_features,num_classes=num_classes,num_layers=num_layers, corr_w = 0.5)

    src_optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.src_cost)
    tgt_optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.tgt_cost)
    #corr_optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.corr_cost)

    task_no = 4
    update_corr = True

    # Assume all tasks are unrelated
    feed_site_corr = np.identity(task_no, dtype='float')/task_no
    sigma_feed_site_corr = np.identity(task_no, dtype='float') / task_no

    print ('feed_site_corr min %.3f, max %.3f'%(feed_site_corr.min(),feed_site_corr.max()))
    feed_site_corr = np.linalg.pinv(feed_site_corr)
    print ('feed_site_corr min %.3f, max %.3f'%(feed_site_corr.min(),feed_site_corr.max()))

    # Due to GTX 970 memory issues
    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True

    # Initialize all variables
    init = tf.global_variables_initializer()

    # model(parameters) saver
    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=config) as sess:
        ##train_summary_writer = tf.summary.FileWriter("C:/tf_logs/train", sess.graph)
        sess.run(init)
        acc_list = []
        f1_list = []
        #tgt_mat = []
        current_max = 0

        print("=" * 50)
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")

            src_list = list(zip(src_total_s1, src_total_s2,src_train_data.labels))
            tgt_list = list(zip(tgt_total_s1, tgt_total_s2,tgt_train_data.labels))

            np.random.shuffle(src_list)
            np.random.shuffle(tgt_list)

            new_src_total_s1, new_src_total_s2, new_src_label = [], [], []
            new_tgt_total_s1, new_tgt_total_s2, new_tgt_label = [], [], []

            for s_tuple in src_list:
                new_src_total_s1.append(s_tuple[0])
                new_src_total_s2.append(s_tuple[1])
                new_src_label.append(s_tuple[2])

            for s_tuple in tgt_list:
                new_tgt_total_s1.append(s_tuple[0])
                new_tgt_total_s2.append(s_tuple[1])
                new_tgt_label.append(s_tuple[2])

            new_src_total_s1 = np.asarray(new_src_total_s1)
            new_src_total_s2 = np.asarray(new_src_total_s2)
            new_src_label = np.asarray(new_src_label)
            new_tgt_total_s1 = np.asarray(new_tgt_total_s1)
            new_tgt_total_s2 = np.asarray(new_tgt_total_s2)
            new_tgt_label = np.asarray(new_tgt_label)

            src_train_data.reset_index()
            tgt_train_data.reset_index()
            m = 0

            while src_train_data.is_available():
                m += 1

                src_batch_x1, src_batch_x2, src_batch_y, src_batch_features = src_train_data.next_batch(new_src_total_s1,new_src_total_s2, labels=new_src_label, batch_size = batch_size)

                if not tgt_train_data.is_available():
                    tgt_train_data.reset_index()

                    # for updating the correlatio matrix
                    if update_corr:
                        if e <= -1:
                            FeedCorr = 0  # 0:skip update, 1:MT (1/2), 2:SGD (no tr=1)
                        else:
                            FeedCorr = 1
                        # update feed_site_corr
                        if FeedCorr == 0:
                            print("Skip update!")
                        else:
                            site_corr_trans, m_w = sess.run(
                                [model.site_corr_trans, model.weights],
                                feed_dict={model.x1: tgt_batch_x1,
                                           model.x2: tgt_batch_x2,
                                           model.y: tgt_batch_y})
                            if FeedCorr == 1:
                                # this part is for the update of omega without logdet
                                if not np.isnan(np.sum(m_w)):
                                    U, s, V = np.linalg.svd(site_corr_trans)
                                    # S = np.sqrt(np.abs(np.diag(s)))
                                    S = np.sqrt(np.diag(s))
                                    A = np.dot(U, np.dot(S, V))
                                    A = A / np.trace(A) # this is the covariance matrix
                                    # print ('  feed_site_corr A(no inv)\n', A)
                                    print_matrix(A, '  feed_site_corr A(no inv)')
                                    print('  feed_site_corr trace', np.trace(A))
                                    # site_corr_new = np.linalg.pinv(A)
                                    # renorm A
                                    B = re_norm(A) # B is the correlation matrix
                                    # print ('  feed_site_corr B(no inv)\n', B)
                                    print_matrix(B, '  feed_site_corr B(no inv)')
                                    site_corr_new = np.linalg.pinv(A) # this is the inverse of the covariance matrix
                                else:
                                    site_corr_new = np.nan
                                    print('m_w nan, skip!!')
                            else:
                                # this part is for the update of omega with logdet, and it may not work
                                site_corr_new = np.linalg.inv(site_corr_trans + sigma_feed_site_corr)
                                # trick
                                site_corr_new = site_corr_new / np.trace(site_corr_new)
                            if np.isnan(np.sum(site_corr_new)):
                                print(' site_corr_new nan, skipped!')
                            else:
                                feed_site_corr = site_corr_new
                            # print (' site_corr_trans', site_corr_trans)
                            if m_w.shape[1] == 1:
                                print(' m_w\t' + " ".join(["%.5f" % (v) for v in m_w]))
                        pass
                    pass

                    # for testing
                    test_data.reset_index()
                    predict_score = []
                    predict_value = []
                    true_score = []
                    QA_pairs = {}
                    #tgt_mat_single = []
                    # labels = test_data.labels
                    s1s, s2s, labels, features = test_data.next_batch(test_total_s1, test_total_s2,
                                                                      labels=test_data.labels,
                                                                      batch_size=test_data.data_size)
                    for i in range(test_data.data_size):
                        pred, clf_input,clf_mat = sess.run([model.tgt_prediction, model.tgt_output_features,model.tgt_att_mat],
                                                   feed_dict={model.x1: np.expand_dims(s1s[i], axis=0),
                                                              model.x2: np.expand_dims(s2s[i], axis=0),
                                                              model.y: np.expand_dims(labels[i], axis=0),
                                                              model.features: np.expand_dims(features[i], axis=0)})

                        true_score.append(labels[i])
                        predict_score.append(np.argmax(pred))
                        '''
                        if i==9 or i==16 or i==23 or i==530:
                            tgt_mat_single.append(clf_mat[:30])
                        '''
                        # print(len(QA_pairs.keys()))
                    #tgt_mat.append(tgt_mat_single)
                    test_acc = test_accuracy(predict_score, true_score)
                    test_f1 = F_score1(predict_score, true_score)
                    acc_list.append(test_acc)
                    '''
                    if test_acc>current_max and test_acc>0.8:
                        fout = open(str(args.t)[14:-4]+'_sentence_test.txt', 'a+')
                        for i in range(test_data.data_size):
                            fout.write('--------------------------'+str(i)+'--------------------------\n')
                            fout.write(str(test_data.s1s[i]))
                            fout.write('\n')
                            fout.write(str(test_data.s2s[i]))
                            fout.write('\n')
                            fout.write('true label: '+str(true_score[i])+'    ')
                            fout.write('predict label: '+str(predict_score[i]))
                            fout.write('\n')
                        fout.close()
                    '''  
                    current_max = max(acc_list)
                    print('acc' + str(test_acc))
                    print("current max acc=" + str(current_max))
                    f1_list.append(test_f1)
                    print('f1' + str(test_f1))
                    print("current max f1=" + str(max(f1_list)))

                tgt_batch_x1, tgt_batch_x2, tgt_batch_y, tgt_batch_features = tgt_train_data.next_batch(
                    new_tgt_total_s1, new_tgt_total_s2, labels = new_tgt_label, batch_size=batch_size)

                '''
                src_batch_x1, src_batch_x2, src_batch_y, src_batch_features = src_train_data.next_batch(src_total_s1,
                                                                                                        src_total_s2, batch_size = batch_size)

                if not tgt_train_data.is_available():
                    tgt_train_data.reset_index()
                tgt_batch_x1, tgt_batch_x2, tgt_batch_y, tgt_batch_features = tgt_train_data.next_batch(
                    tgt_total_s1, tgt_total_s2, batch_size=batch_size)
                '''

                _, sc = sess.run([src_optimizer, model.src_cost],
                                                  feed_dict={model.x1: src_batch_x1,
                                                             model.x2: src_batch_x2,
                                                             model.y: src_batch_y,
                                                             model.site_corr: feed_site_corr,
                                                             model.features: src_batch_features})

                _, tc = sess.run([tgt_optimizer, model.tgt_cost],
                                                  feed_dict={model.x1: tgt_batch_x1,
                                                             model.x2: tgt_batch_x2,
                                                             model.y: tgt_batch_y,
                                                             model.site_corr: feed_site_corr,
                                                             model.features: tgt_batch_features})
                
                #if e > 5:
                   # _, cc = sess.run([corr_optimizer, model.corr_cost],
                    #                              feed_dict={model.site_corr: feed_site_corr})
                #else:
                    #cc = 0
                                                            
                if m % 100 == 0:
                    print("[batch " + str(m) + "] src cost:", sc)
                    print("[batch " + str(m) + "] tgt cost:", tc)
                    #_, cc = sess.run([corr_optimizer, model.corr_cost],
                     #                             feed_dict={model.site_corr: feed_site_corr})
                    #print("[batch " + str(i) + "] corr cost:", cc)
                    
                #train_summary_writer.add_summary(merged, i)


        print("training finished!")
        print("=" * 50)
        


if __name__ == "__main__":
#def run():
    src = pd.read_csv(args.s)
    src.dropna(inplace=True)
    tgt = pd.read_csv(args.t)
    tgt.dropna(inplace=True)
    tgtrain,tgtest = train_test_split(tgt)
    params = {
        "lr": 0.08,
        "ws": 4,
        "l2_reg": 0.0004,
        "epoch": 5,
        "batch_size": 64,
        "model_type": "Model31",
        "num_layers": 1,
        "data_type": "AliExp"
    }

    print("=" * 50)
    print("Parameters:")
    for k in sorted(params.keys()):
        print(k, ":", params[k])
    train(src,tgtrain,tgtest,lr=float(params["lr"]), w=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
          batch_size=int(params["batch_size"]), model_type=params["model_type"], num_layers=int(params["num_layers"]),
          data_type=params["data_type"])

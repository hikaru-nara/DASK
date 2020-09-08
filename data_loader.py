import os, numpy as np
import scipy.sparse as sp
from gensim.corpora import Dictionary as gensim_dico
from brain.knowgraph import KnowledgeGraph


class bdek_train_dataset(torch.utils.data.IterableDataset):
    def __init__(self, source_name, target_name, graph_path max_words=5000, supervision_rate=0.1, repeat=1):
        super(bdek_dataset, self).__init__()
        X_s, Y_s, d_s, g_s, self.X_tu, self.Y_tu, self.d_tu, self.g_tu, X_semi, Y_semi, d_semi, g_semi, \
            _,_,_,_ = forge_dataset(source_name, target_name, max_words, supervision_rate)
        # print(X_s.shape)
        self.X_l = np.concatenate((X_s, X_semi))
        self.Y_l = np.concatenate((Y_s, Y_semi))
        self.d_l = np.concatenate((d_s, d_semi))
        self.g_l = np.concatenate((g_s, g_semi))
        self.kg = KnowledgeGraph(graph_path, predicate)

def create_vocab(sentence_list):
    for sent in sentence_list:
        for word in 

def forge_dataset(source_name, target_name, max_words, supervision_rate):
    t_s, y_s = get_labeled_dataset(source_name)
    d_s = np.zeros(len(t_s))
    t_tu = get_unlabeled_dataset(target_name)
    d_tu = np.ones(len(t_tu))
    t_tl, y_tl = get_labeled_dataset(target_name)
    d_tl = np.ones(len(t_tl))
    g_s, g_tu, g_tl = get_graph_feature(source_name, target_name)
    vocab = create_vocab(t_s+t_tu+t_tl)


def get_graph_feature(source_name, target_name):
    '''
    get the graph features from source and target dataset
    '''
    g_s = np.load(open('graph_features/sf_' + d1 +'_small_5000.np', 'rb'), allow_pickle=True)
    g_tu = np.load(open('graph_features/sf_' + d2 + '_small_5000.np', 'rb'), allow_pickle=True)
    g_tl = np.load(open('graph_features/sf_'+ d2 + '_test_5000.np', 'rb'), allow_pickle=True)
    return g_s, g_tu, g_tl


###########################


def parse_processed_amazon_dataset(FNames, max_words):
    datasets = {}
    dico = gensim_dico()

    # First pass on document to build dictionary
    for fname in FNames:
        f = open(fname)
        for l in f:
            tokens = l.split(sep=' ')
            label_string = tokens[-1]
            tokens_list=[]
            for tok in tokens[:-1]:
                ts, tfreq = tok.split(':')
                freq = int(tfreq)
                tokens_list += [ts]*freq

            _ = dico.doc2bow(tokens_list, allow_update=True)

        f.close()

    # Preprocessing_options
    dico.filter_extremes(no_below=2, keep_n=max_words)
    dico.compactify()

    for fname in FNames:
        X = []
        Y = []
        docid = -1
        f = open(fname)
        for l in f:
            tokens = l.split(sep=' ')
            label_string = tokens[-1]
            tokens_list = []
            for tok in tokens[:-1]:
                ts, tfreq = tok.split(':')
                freq = int(tfreq)
                tokens_list += [ts]*freq

            count_list = dico.doc2bow(tokens_list, allow_update=False)

            docid += 1

            X.append((docid, count_list))

            # Preprocess Label
            ls, lvalue = label_string.split(':')
            if ls == "#label#":
                if lvalue.rstrip() == 'positive':
                    lv = 1
                    Y.append(lv)
                elif lvalue.rstrip() == 'negative':
                    lv = 0
                    Y.append(lv)
                else:
                    raise Exception("Invalid Label Value")
            else:
                raise Exception('Invalid Format')

        datasets[fname] = (X, np.array(Y))
        f.close()
        del f

    return datasets, dico


def count_list_to_sparse_matrix(X_list, dico):
    ndocs = len(X_list)
    voc_size = len(dico.keys())


    X_spmatrix = sp.lil_matrix((ndocs, voc_size))
    for did, counts in X_list:
        for wid, freq in counts:
            X_spmatrix[did, wid]=freq

    return X_spmatrix.tocsr()


def get_dataset_path(domain_name, exp_type):
    prefix ='./dataset/'
    if exp_type == 'small':
        fname = 'labelled.review'
    elif exp_type == 'all':
        fname = 'all.review'
    elif exp_type == 'test':
        fname = 'unlabeled.review'

    return os.path.join(prefix, domain_name, fname)

def split_semi_supervision(texts, labels, supervision_rate=0.1):
    
    return (None, None, None, None)


def get_dataset(source_name, target_name, max_words=5000):

    source_path  = get_dataset_path(source_name, 'small')
    target_path1 = get_dataset_path(target_name, 'small')
    target_path2 = get_dataset_path(target_name, 'test')

    dataset_list = [source_path, target_path1, target_path2]
    datasets, dico = parse_processed_amazon_dataset(dataset_list, max_words, supervision_rate)

    L_s, Y_s = datasets[source_path]
    L_tu, Y_tu = datasets[target_path1]
    L_tl, Y_tl = datasets[target_path2]

    # L_semi, Y_semi, L_test, Y_test = split_semi_supervision(L_t2, Y_t2)

    X_s  = count_list_to_sparse_matrix(L_s,  dico)
    X_tu = count_list_to_sparse_matrix(L_tu, dico)
    X_tl = count_list_to_sparse_matrix(L_tl, dico)

    d_s = np.zeros(len(L_s)).astype(np.int64)
    d_tu = np.ones(len(L_tu)).astype(np.int64)
    d_tl = np.ones(len(L_tl)).astype(np.int64)

    return X_s, Y_s, d_s, X_tu, Y_tu, d_tu, X_tl, Y_tl, d_tl, dico

def get_graph_feature(source_name, target_name):
    '''
    get the graph features from source and target dataset
    '''
    g_s = np.load(open('graph_features/sf_' + d1 +'_small_5000.np', 'rb'), allow_pickle=True)
    g_tu = np.load(open('graph_features/sf_' + d2 + '_small_5000.np', 'rb'), allow_pickle=True)
    g_tl = np.load(open('graph_features/sf_'+ d2 + '_test_5000.np', 'rb'), allow_pickle=True)
    return g_s, g_tu, g_tl

def get_commonsense_graph(graph_path, predicate=True):
    return KnowledgeGraph(graph_path, predicate)

def forge_dataset(source_name, target_name, max_words=5000, supervision_rate=0.1):
    g_s, g_tu, g_tl = get_graph_feature(source_name, target_name)
    X_s, Y_s, d_s, X_tu, Y_tu, d_tu, X_tl, Y_tl, d_tl, _ = get_dataset(source_name, target_name, max_words)
    X_semi, Y_semi, d_semi, g_semi, X_te, Y_te, d_te, g_te = split_semi_supervision(X_tl, Y_tl, d_tl, g_tl)
    return X_s, Y_s, d_s, g_s, X_tu, Y_tu, d_tu, g_tu, X_semi, Y_semi, d_semi, g_semi, X_te, Y_te, d_te, g_te

if __name__=='__main__':
    D_tr = bdek_train_dataset('books','dvd')


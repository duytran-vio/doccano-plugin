from Scripts.utils import *

def read_from_file(data_path, corpus_path):
    ###GET DATA FROM FILE
    part = pd.read_csv(data_path)
    part = part.dropna()
    full = pd.read_csv(corpus_path, sep='\n', header=None)
    full = full.dropna()

    return part, full

def data_from_file(part, full, text_col, label_col):

    part = equalize_label(part, label_col)
    ###RETRIEVE COLS
    X = part[:][text_col].values  
    y = part[:][label_col].values

    X_corp = full.values

    ###PREPROCESS DATA
    data = preprocess(X)
    corpus = preprocess(X_corp, remove_empty=True)

    return data, y, corpus

def svm_data_prepair(X_tfidf, label):
    indexed = np.linspace(0, label.shape[0]-1, label.shape[0])
    y_idx = np.concatenate( [indexed.reshape([-1,1]), label.reshape([-1,1]) ] , axis = 1)

    X_train, X_test, y_idx_train, y_idx_test = train_test_split(X_tfidf, y_idx, test_size=0.25, random_state=True)
    y_train = y_idx_train[:,1]
    y_test = y_idx_test[:,1]

    return X_train, X_test, y_train, y_test, y_idx_train, y_idx_test
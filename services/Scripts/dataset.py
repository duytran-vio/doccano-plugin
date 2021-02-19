from Scripts.utils import *

def data_from_file(data_path, text_col, label_col):
    ###GET DATA FROM FILE
    part = pd.read_excel(data_path)[[text_col, label_col]]
    # part = part.dropna()
    # part = equalize_label(part, label_col)

    ###RETRIEVE COLS
    X = part[:][text_col].values  
    y = part[:][label_col].values

    ###PREPROCESS DATA
    data = preprocess(X)
    label = [0 if type(i) is float else 1 for i in y]

    return data, label

def corpus_from_file(corpus_path):
    full = pd.read_csv(corpus_path, sep='\n', header=None)
    full = full.dropna()

    X_corp = full.values
    corpus = preprocess(X_corp, remove_empty=True)

    return corpus, full

def svm_data_prepair(X_tfidf, label, test_size):
    indexed = np.linspace(0, label.shape[0]-1, label.shape[0])
    y_idx = np.concatenate( [indexed.reshape([-1,1]), label.reshape([-1,1]) ] , axis = 1)

    X_train, X_test, y_idx_train, y_idx_test = train_test_split(X_tfidf, y_idx, test_size=test_size, random_state=True)
    y_train = y_idx_train[:,1]
    y_test = y_idx_test[:,1]

    return X_train, X_test, y_train, y_test, y_idx_train, y_idx_test
from sklearn.feature_extraction.text import TfidfVectorizer 
import pickle

def make_tfidf_model(corpus, save_path = None, pretrained_path = None):
    if pretrained_path is None:
        tfidfconverter = TfidfVectorizer(max_features=10000)  
        tfidfconverter = tfidfconverter.fit(corpus)
        if save_path is not None:
            pickle.dump(tfidfconverter, open(save_path, 'wb'))
            print("Model is saved to ", save_path)
    else:
        tfidfconverter = pickle.load(open(pretrained_path, 'rb'))
        print("Model is loaded from ", pretrained_path)

    return tfidfconverter    
    
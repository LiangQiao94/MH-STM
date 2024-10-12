import numpy as np
import pandas as pd
import random,nltk,os
import gensim
import gzip
import scipy
import _pickle as cPickle
import logging
from MHSTM_model import NCRPNode, HierarchicalLDA
from sklearn.preprocessing import LabelEncoder
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)      

    
    
def evaluate(model, leaf_label, brand_aspect):
    # leaf prediction accuracy
    y_true = []
    for line in open(leaf_label): 
        y_true.append(int(line.strip()))  
    
    y_pred = []
    for d in range(len(model.corpus)):
        for s in range(len(model.corpus[d])):
            node = model.document_leaves[d][s]
            y_pred.append(node.node_id)
    encoder = LabelEncoder()
    y_pred = encoder.fit_transform(y_pred)        
                
    dim = len(np.unique(y_true))
    w = np.zeros((dim, dim))
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    # match the predicted topics with the ground truth by Hungarian Algorithm
    from scipy.optimize import linear_sum_assignment
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    leaf_acc = sum([w[i, j] for i, j in ind]) / len(y_pred)
 
    # multi-aspect brand ranking
    brand_aspect_true = []
    for line in open(brand_aspect): 
        brand_aspect_true.append(line.strip().split('\t'))
    brand_aspect_true = np.array(brand_aspect_true, dtype = np.float)
    
    model.topics = model.search_topics(model.root_node)
    leaf_topics = [topic for topic in model.topics if topic.level == model.num_levels-1]
    leaf_topics_id = [topic.node_id for topic in leaf_topics]
    encoded_y = encoder.transform(leaf_topics_id)
    spearmanr = []
    kendalltau = []
    APs = []
    for i in range(len(leaf_topics)):
        matched_aspect = ind[encoded_y[i]][1]
        score_pred = leaf_topics[i].omega + leaf_topics[i].parent.omega + leaf_topics[i].parent.parent.omega
        spearmanr.append(scipy.stats.spearmanr(score_pred,brand_aspect_true[matched_aspect])[0])
        kendalltau.append(scipy.stats.kendalltau(score_pred,brand_aspect_true[matched_aspect])[0])
        APs.append(topK_AP(score_pred, brand_aspect_true[matched_aspect], K = 5))  
    avg_spearmanr = np.mean(spearmanr)
    avg_kendalltau = np.mean(kendalltau)
    avg_AP = np.mean(APs)
    
    print(leaf_acc,'\t',avg_spearmanr,'\t',avg_kendalltau,'\t',avg_AP)   
    return     

def topK_AP(regression_coef,brand_aspect,K=10):
    topK_brand_idx = np.argsort(brand_aspect)[::-1][:K]
    predict_brand_idx = np.argsort(regression_coef)[::-1][:K]
    N_hits = 0    
    score = 0
    n = 0
    for i in range(K):
        if predict_brand_idx[i] in topK_brand_idx:
            N_hits += 1
        score += N_hits/(i+1)
        n += 1
    return score/n

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)
        
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

def load_data(data_dir):
    
    with open(data_dir,encoding='utf-8') as fcorpus:
	    raw_reviews = fcorpus.read().splitlines()
    reviews = [content.split('\t')[0] for content in raw_reviews]
    overall = [float(content.split('\t')[1]) for content in raw_reviews]    
    brands = [int(content.split('\t')[2]) for content in raw_reviews]
    docs = []
    plain_texts = []
    for i in range(len(reviews)): 
        sentences = []
        plain_text = []
        for doc in reviews[i].split('.'):
            doc = nltk.word_tokenize(doc)
            sentences.append(doc)
            plain_text += doc
        docs.append(sentences)
        plain_texts.append(plain_text)
      
    dictionary = gensim.corpora.Dictionary(plain_texts)
    corpus = []
    corpus_brands = []
    corpus_ratings = []
    brand_idx = [0]
    # pack into corpus
    for i in range(len(docs)):
        doc = []
        for sentence in docs[i]:
            words = [dictionary.token2id[word] for word in sentence if word in dictionary.token2id.keys()]
            if len(words)>0:
                doc.append(words)
        if len(doc)>0:
            corpus.append(doc)
            corpus_brands.append(brands[i])
            corpus_ratings.append(overall[i])
            if i>0 and corpus_brands[-2] != corpus_brands[-1]:
                brand_idx.append(len(corpus_brands)-1)
    brand_idx.append(len(corpus_brands))
    
    return dictionary, corpus, corpus_brands, corpus_ratings, brand_idx



if __name__ == "__main__":
    
    random.seed(0)
    
    # simulation path
    branches = [3,2,4]
    eta = 1.0  # smoothing over topic-word distributions
    data_dir = './data/simulation/simulated_corpus_'+'-'.join([str(ele) for ele in branches]+[str(eta)]) + '.txt'
    leaf_label = './data/simulation/leaf_label_'+'-'.join([str(ele) for ele in branches]+[str(eta)]) + '.txt'
    brand_aspect = './data/simulation/brand_ranking_'+'-'.join([str(ele) for ele in branches]+[str(eta)]) + '.txt'
    exp_dir = './models/simulation/'+'-'.join([str(ele) for ele in branches]+[str(eta)])
    if not os.path.isdir(exp_dir):
    	os.mkdir(exp_dir)
       
    # load data
    dictionary, corpus, corpus_brands, corpus_ratings, brand_idx = load_data(data_dir)  
   
    # modeling
    alpha = 1.0           # smoothing over level distributions
    num_levels = 3        # the number of levels in the tree
    brand_num = 10        # the number of brands
    
    # repeat 20 times
    models = [0]*20
    for i in range(20):
        print("model No. ",i+1)
        models[i] = HierarchicalLDA(corpus=corpus, ratings=corpus_ratings, brands = corpus_brands, vocab=dictionary, brand_num=brand_num, brand_idx=brand_idx, alpha=alpha, eta=eta, num_levels=num_levels)    
        models[i].init_assign(branches=branches)
        models[i].estimate(num_iters=50, display_topics=10, n_words=10, fix_tree=True)
        # save model for future use 
        save_zipped_pickle(models[i], exp_dir + '/MHSTM_'+str(i+1)+'.p')

    # load saved models for evaluation
    models = [0]*20
    for i in range(20): 
        with gzip.open(exp_dir + '/MHSTM_'+str(i+1)+'.p', 'rb') as f:
            models[i] = cPickle.load(f)
    
    # hierarchical affinity
    Affinities = []
    for i in range(20):
        Affinities.append(models[i].Affinity())
    print(np.mean(Affinities),np.std(Affinities,ddof=1)/np.sqrt(20))
    
    # likelihood
    likelihoods = []
    for i in range(20):
        likelihoods.append(models[i].word_likelihood())
    print(np.mean(likelihoods),np.std(likelihoods,ddof=1)/np.sqrt(20))

    # topic accuracy & multi-aspect brand ranking accuracy
    for i in range(20):
        try:
            evaluate(models[i], leaf_label, brand_aspect)
        except ValueError:
            print("skip")
    
    # coherence  
    coherence_scores = []
    for i in range(20):
        coherence_scores.append(models[i].coherence(5))
    print(np.mean([score[0] for score in coherence_scores]))
    print(np.std([score[0] for score in coherence_scores], ddof = 1)/np.sqrt(20))
    
    
    '''
    # demo path
    data_dir = './data/demo/simulated_corpus.txt'
    leaf_label = './data/demo/leaf_label.txt'
    brand_aspect = './data/demo/brand_ranking.txt'
    exp_dir = './models/demo'
    if not os.path.isdir(exp_dir):
    	os.mkdir(exp_dir)
       
    # load demo data
    dictionary, corpus, corpus_brands, corpus_ratings, brand_idx = load_data(data_dir)  
    
    # modeling for demo
    alpha = 1.0           # smoothing over level distributions
    num_levels = 3        # the number of levels in the tree
    brand_num = 10        # the number of brands
    eta = 0.01
    branches = [3,3,3]
    model = HierarchicalLDA(corpus=corpus, ratings=corpus_ratings, brands = corpus_brands, vocab=dictionary, brand_num=brand_num, brand_idx=brand_idx, alpha=alpha, eta=eta, num_levels=num_levels)    
    model.init_assign(branches=branches)
    model.estimate(num_iters=50, display_topics=10, n_words=10, fix_tree=True)
    #save_zipped_pickle(model, exp_dir + '/MHSTM_demo.p')
    
    # visualize topics in demo
    import seaborn as sns
    with gzip.open(exp_dir + '/MHSTM_demo.p', 'rb') as f:
        model = cPickle.load(f)
    t = model.root_node.children[2].children[0]
    density = t.word_counts/np.sum(t.word_counts)
    phi = np.zeros(9)
    for i in range(9):
        idx = model.vocab.token2id[str(i)]
        phi[i] = density[idx]
    mat = phi.reshape((3,3))
    sns.heatmap(mat,vmin=0,vmax=1,cmap='Blues',xticklabels=False,yticklabels=False,cbar=True,square=True,linewidths=0.5) 
    '''
   
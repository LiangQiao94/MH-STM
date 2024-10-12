from math import log
from MHSTM_model import NCRPNode, HierarchicalLDA
from numpy.random import RandomState
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import random,nltk,os
import string
import gensim
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import gzip
import scipy
import _pickle as cPickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def aspect_estimate(model, num_iters, aspect_corpus):
    '''
    estimation for hold-out aspect corpus with a given model
    '''
    regression_coef = np.zeros(model.brand_num)
    path = np.zeros(model.num_levels, dtype=np.object)
    for d in range(len(aspect_corpus)):
        #logging.info('review ' + str(d+1))
        doc = aspect_corpus[d]
        # initialize document topic assignments
        path[0] = model.root_node 
        for level in range(1, model.num_levels):                    
            parent_node = path[level-1]
            level_node = parent_node.select_exist()
            path[level] = level_node
        # set the leaf node for this document
        doc_leaf = path[model.num_levels-1]               
        # randomly assign each word in the document to a level (node) along the path
        doc_levels = np.zeros(len(doc), dtype=np.int)
        for n in range(len(doc)):
            random_level = model.random_state.randint(model.num_levels)
            doc_levels[n] = random_level
           
        # gibbs sampling of path and levels for the doc
        for iteration in range(num_iters):         
            doc_leaf = aspect_sample_path(model, doc, doc_levels, doc_leaf)
            aspect_sample_topics(model, doc, doc_levels, doc_leaf)
            
        # get the leaf node and populate the path
        node = doc_leaf
        for level in range(model.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
            path[level] = node
            node = node.parent
        for level in doc_levels:
            topic = path[level]
            for brand in range(model.brand_num):
                regression_coef[brand] += topic.omega[brand]                     
                if level == 1:
                    regression_coef[brand] += topic.parent.omega[brand] 
                if level == 2:
                    regression_coef[brand] += (topic.parent.omega[brand] + topic.parent.parent.omega[brand] )
                    
    return regression_coef 


def aspect_sample_path(model, doc, levels, leaf):
           
    # define a path starting from the leaf node of this doc
    path = np.zeros(model.num_levels, dtype=np.object)
    node = leaf
    for level in range(model.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
        path[level] = node
        node = node.parent

    # calculates the prior p(c_d | c_{-d}) in eq. (4)
    total_topic_nodes = model.search_topics(model.root_node)

    leaf_weights = {}
    for node in total_topic_nodes:
        if node.is_leaf() and node.customers>0:
            leaf_weights[node] = log(float(node.customers)/model.root_node.customers)


    # calculates the likelihood p(w_d,y_d | c, w_{-d}, z) in eq. (4)
    level_word_counts = {}
    for level in range(model.num_levels):
        level_word_counts[level] = {}

    # remove doc from path
    for n in range(len(doc)): # for each word in the doc
        # count the word at each level
        level = levels[n]
        w = doc[n]
        if w not in level_word_counts[level]:
            level_word_counts[level][w] = 1
        else:
            level_word_counts[level][w] += 1


    # likelihood of w_d and y_d
    aspect_calculate_likelihood(model, leaf_weights, model.root_node, 0.0, level_word_counts, 0)

    # pick a new path
    nodes = np.array(list(leaf_weights.keys()))
    weights = np.array([leaf_weights[node] for node in nodes])
    weights = np.exp(weights - np.max(weights)) # normalise so the largest weight is 1
    weights = weights / np.sum(weights)

    choice = model.random_state.multinomial(1, weights).argmax()
    leaf_node = nodes[choice]
                
    return leaf_node
           
    
def aspect_calculate_likelihood(model, leaf_weights, node, parent_weight, level_word_counts, level):

    # calculate the likelihood of the words at this level, given this topic
    node_weight = parent_weight
    word_counts = level_word_counts[level]

    for w in word_counts:
        node_weight += word_counts[w]* log( (model.eta + node.word_weighted_counts[w] ) /
                                    (model.eta_sum + node.total_weighted_counts ) )        
       
    # propagate that weight to the child nodes
    for child in node.children:
        aspect_calculate_likelihood(model, leaf_weights, child, node_weight, level_word_counts, level+1)
    if node.is_leaf() and node in leaf_weights.keys():
        leaf_weights[node] += node_weight  # word likelihood
           

def aspect_sample_topics(model, doc, levels, leaf):

    # initialise level counts
    level_counts = np.zeros(model.num_levels, dtype=np.int)
    for z in levels:
        level_counts[z] += 1

    # get the leaf node and populate the path
    path = np.zeros(model.num_levels, dtype=np.object)
    node = leaf
    for level in range(model.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
        path[level] = node
        node = node.parent

    # sample a new level for each word
    level_weights = np.zeros(model.num_levels)
    for n in range(len(doc)):
        w = doc[n]
        word_level = levels[n]

        # remove from model
        level_counts[word_level] -= 1

        # pick new level
        for level in range(model.num_levels):
            level_weights[level] = (model.alpha + level_counts[level]) *                    \
                        (model.eta + path[level].word_weighted_counts[w]) / (model.eta_sum + path[level].total_weighted_counts)
        level_weights = level_weights / np.sum(level_weights)
        level = model.random_state.multinomial(1, level_weights).argmax()

        # put the word back into the model
        levels[n] = level
        level_counts[level] += 1

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




if __name__ == "__main__":
    
    random.seed(0)
    
    data_dir = './data/ratebeer_2010.txt'
    
    # load model
    with gzip.open('./models/MHSTM_beer.p', 'rb') as f:
        model = cPickle.load(f)

    
    # load aspect corpus
    aspect_dir = './data/beer_aspect/beer_appearance.txt'       
    with open(aspect_dir,encoding='utf-8') as fcorpus:
	    reviews = fcorpus.read().splitlines()  
    
    stemmer = PorterStemmer()
    stopset = stopwords.words('english') + ['will', 'also', 'said']
    del_str = string.punctuation + string.digits
    replace = str.maketrans(del_str,' '*len(del_str))
    docs = []
    for i in range(len(reviews)):
        doc = reviews[i]
        # strip punctuations and digits
        doc = doc.translate(replace) 
        doc = doc.encode("utf8").decode("utf8").encode('ascii', 'ignore').decode() # ignore fancy unicode chars  
        doc = nltk.word_tokenize(doc)
        doc = [w.lower() for w in doc]
        doc = [w for w in doc if w not in stopset]
        doc = [stemmer.stem(w) for w in doc]
        doc = [model.vocab.token2id[w] for w in doc if w in model.vocab.token2id.keys()]       
        docs.append(doc)


    # compute brand average aspect ratings   
    data = pd.read_table(data_dir,sep='\t',header=None)
    data.columns = ['reviews','overall','appearance','aroma','palate','taste','brands']
    
    brand_aspect = []
    brand_overall= []
    for brand in range(1,27):
        brand_data = data[data['brands']==brand]
        brand_aspect.append(brand_data['appearance'].mean())
        brand_overall.append(brand_data['overall'].mean())
    brand_aspect = np.array(brand_aspect)
    brand_overall = np.array(brand_overall)
    
    # compute correlation between brand aspect ratings and corresponding regression parameters
    regression_coef = aspect_estimate(model, num_iters=10, aspect_corpus=docs)
    
    
    print(scipy.stats.spearmanr(regression_coef,brand_aspect)[0])
    print(scipy.stats.kendalltau(regression_coef,brand_aspect)[0])
    print(topK_AP(regression_coef, brand_aspect, K=10))
    print(topK_AP(regression_coef, brand_aspect, K=5))
    
    

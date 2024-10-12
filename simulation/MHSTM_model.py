import math
from math import log
from math import exp
import statsmodels.api as sm
from numpy.random import RandomState
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import random
import nltk
import os
import string
import gensim
from gensim.models import CoherenceModel
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import gzip
import scipy
import _pickle as cPickle
import logging
from sklearn.utils import shuffle
#from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import KFold
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class NCRPNode(object):

    # class variable to keep track of total nodes created so far
    total_nodes = 0
    last_node_id = 0

    def __init__(self, num_levels, vocab, brand_num, num_documents, parent=None, level=0,
                 random_state=None):

        self.node_id = NCRPNode.last_node_id
        NCRPNode.last_node_id += 1

        self.customers = 0
        self.brand_customers = np.zeros(brand_num)
        self.parent = parent
        self.children = []
        self.level = level
        self.num_levels = num_levels
        self.brand_num = brand_num
        self.omega = np.zeros(brand_num)     # regression parameter of each node
        self.vocab = vocab
        self.word_counts = np.zeros(len(vocab))
        self.word_weighted_counts = np.zeros(len(vocab))
        self.total_weighted_counts = 0
        self.word_last_weights = np.ones(len(vocab))
        self.word_cur_weights = np.ones(len(vocab))
        self.num_documents = num_documents
        self.z_d = np.zeros(num_documents)
        self.brand_word_counts = np.zeros((brand_num, len(vocab)))

        if random_state is None:
            self.random_state = RandomState()
        else:
            self.random_state = random_state

    def __repr__(self):
        parent_id = None
        if self.parent is not None:
            parent_id = self.parent.node_id
        return 'Node=%d level=%d customers=%d total_weighted_counts=%d parent=%s' % (self.node_id,
                                                                                     self.level, self.customers, self.total_weighted_counts, parent_id)

    def add_child(self):
        ''' Adds a child to the next level of this node '''
        node = NCRPNode(self.num_levels, self.vocab, self.brand_num,
                        self.num_documents, parent=self, level=self.level+1)
        self.children.append(node)
        NCRPNode.total_nodes += 1
        return node

    def is_leaf(self):
        ''' Check if this node is a leaf node '''
        return self.level == self.num_levels-1

    def get_new_leaf(self):
        ''' Keeps adding nodes along the path until a leaf node is generated'''
        node = self
        for l in range(self.level, self.num_levels-1):
            node = node.add_child()
        return node

    def drop_path(self, brand):
        ''' Removes a document from a path starting from this node '''
        node = self
        node.customers -= 1
        node.brand_customers[brand] -= 1
        if node.customers == 0:
            node.parent.remove(node)
        for level in range(1, self.num_levels-1):  # skip the root
            node = node.parent
            node.customers -= 1
            node.brand_customers[brand] -= 1
            if node.customers == 0:
                node.parent.remove(node)

    def remove(self, node):
        ''' Removes a child node '''
        self.children.remove(node)
        NCRPNode.total_nodes -= 1

    def add_path(self, brand):
        ''' Adds a document to a path starting from this node '''
        node = self
        node.customers += 1
        node.brand_customers[brand] += 1
        for level in range(1, self.num_levels-1):  # skip the root
            node = node.parent
            node.customers += 1
            node.brand_customers[brand] += 1

    def select(self, gamma):
        ''' Selects an existing child or create a new one according to the CRP '''

        weights = np.zeros(len(self.children)+1)
        weights[0] = float(gamma) / (gamma+self.customers)
        i = 1
        for child in self.children:
            weights[i] = float(child.customers) / (gamma + self.customers)
            i += 1

        choice = self.random_state.multinomial(1, weights).argmax()
        if choice == 0:
            return self.add_child()
        else:
            return self.children[choice-1]

    def select_exist(self):
        ''' Selects an existing child '''

        weights = np.zeros(len(self.children))
        i = 0
        for child in self.children:
            weights[i] = float(child.customers) / (self.customers)
            i += 1

        choice = self.random_state.multinomial(1, weights).argmax()

        return self.children[choice]

    def get_top_words(self, n_words, with_weight):
        ''' Get the top n words in this node '''

        pos = np.argsort(self.word_counts)[::-1]
        sorted_counts = self.word_counts[pos][:n_words]
        sorted_weights = sorted_counts/np.sum(self.word_counts)
        sorted_vocab = [self.vocab[idx] for idx in pos[:n_words]]

        output = ''
        for word, weight in zip(sorted_vocab, sorted_weights):
            if with_weight:
                output += '%s (%.3f), ' % (word, weight)
            else:
                output += '%s, ' % word
        return output


class HierarchicalLDA(object):

    def __init__(self, corpus, ratings, brands, vocab, brand_num, brand_idx,
                 alpha=1.0, gamma=1.0, eta=0.1, rho_square = 1.0,
                 seed=0, verbose=True, num_levels=3):

        NCRPNode.total_nodes = 0
        NCRPNode.last_node_id = 0

        self.background_counts = np.zeros(len(vocab))
        self.corpus = corpus
        self.w_d = np.zeros(len(corpus))     # regression predictions for each review
        self.ratings = np.array(ratings,dtype=np.float)  # ratings for each review
        # zero-mean ratings
        self.ratings -= np.mean(self.ratings)
        self.vocab = vocab
        self.brands = brands
        self.alpha = alpha  # smoothing on doc-topic distributions
        self.gamma = gamma  # "imaginary" customers at the next, as yet unused table
        self.eta = eta      # smoothing on topic-word distributions
        self.rho_square = rho_square
        self.seed = seed
        self.random_state = RandomState(seed)
        self.verbose = verbose
        self.brand_num = brand_num
        self.brand_idx = brand_idx
        self.num_levels = num_levels
        self.num_documents = len(corpus)
        self.document_length = np.zeros(self.num_documents)
        self.eta_sum = eta * len(vocab)
        
        
    def init_assign(self, branches=None):      

        self.root_node = NCRPNode(
            self.num_levels, self.vocab, self.brand_num, self.num_documents)

        # initialize a fixed tree structure if needed
        if branches is not None: 
            self.root_node.customers += sum(branches)
            for i in range(len(branches)):
                node = self.root_node.add_child()
                node.customers += branches[i]
                for j in range(branches[i]):
                    child = node.add_child()
                    child.customers += 1     
             
        # initialise a single path
        path = np.zeros(self.num_levels, dtype=np.object)

        # currently selected path (i.e. leaf node) through the NCRP tree
        self.document_leaves = []
        # indexed < doc, token >
        self.levels = np.zeros(self.num_documents, dtype=np.object)
        #self.brand_levels = np.zeros(self.brand_num)
        for d in range(len(self.corpus)):
            d_leaves = []
            self.levels[d] = np.zeros(len(self.corpus[d]), dtype=np.object)
            # always add to the root node first
            self.root_node.customers += len(self.corpus[d])
            brand = self.brands[d]
            self.root_node.brand_customers[brand] += len(self.corpus[d])
            for s in range(len(self.corpus[d])):
                doc = self.corpus[d][s]
                # populate nodes into the path of this document
                doc_len = len(doc)
                self.document_length[d] += doc_len
                path[0] = self.root_node
                for level in range(1, self.num_levels):
                    # at each level, a node is selected by its parent node based on the CRP prior
                    parent_node = path[level-1]
                    if branches is None:
                        level_node = parent_node.select(self.gamma)
                    else:
                        level_node = parent_node.select_exist()
                    level_node.customers += 1
                    level_node.brand_customers[brand] += 1
                    path[level] = level_node

                # set the leaf node for this document
                leaf_node = path[self.num_levels-1]
                d_leaves.append(leaf_node)

                # randomly assign each word in the document to a level (node) along the path
                self.levels[d][s] = np.zeros(doc_len, dtype=np.int)
                for n in range(doc_len):
                    w = doc[n]
                    random_level = self.random_state.randint(self.num_levels)
                    random_node = path[random_level]
                    self.hpu_add_word_counts(random_node, w, 1)
                    random_node.z_d[d] += 1
                    iter_node = random_node.parent
                    while(iter_node):
                        iter_node.z_d[d] += 1
                        iter_node = iter_node.parent
                    random_node.brand_word_counts[brand, w] += 1
                    self.levels[d][s][n] = random_level
                    self.background_counts[w] += 1
            self.document_leaves.append(d_leaves)

    def hpu_add_word_counts(self, node, w, num):
        node.word_counts[w] += num
        node.word_weighted_counts[w] += num
        node.total_weighted_counts += num
        while (node.parent):
            node = node.parent
            node.word_counts[w] += num
            node.word_weighted_counts[w] += num * node.word_cur_weights[w]
            node.total_weighted_counts += num * node.word_cur_weights[w]

    def hpu_reduce_word_counts(self, node, w, num):
        node.word_counts[w] -= num
        node.word_weighted_counts[w] -= num
        node.total_weighted_counts -= num
        while (node.parent):
            node = node.parent
            node.word_counts[w] -= num
            node.word_weighted_counts[w] -= num * node.word_last_weights[w]
            node.total_weighted_counts -= num * node.word_last_weights[w]

    def search_topics(self, node):
        # recursively search descendant topic nodes under a certain node including itself
        descendants = []
        queue = [node]
        while (len(queue) > 0):
            topic = queue.pop(0)
            descendants.append(topic)
            for child in topic.children:
                queue.append(child)
        return descendants

    def estimate(self, num_iters, display_topics=50, n_words=5, with_weights=False, fix_tree=False):

        self.topics = self.search_topics(self.root_node)
        for iteration in range(num_iters):

            by_brand = True

            # Gibbs sampling E-step
            # - sample path
            #logging.info('sample paths')
            if fix_tree: # keep the current tree structure
                for d in range(len(self.corpus)):
                    for s in range(len(self.corpus[d])):                  
                        self.sample_existing_path(d, s, by_brand)
            else:  # adjust the tree structure by adding or deleting topic nodes
                for d in range(len(self.corpus)):
                    for s in range(len(self.corpus[d])):                       
                        self.sample_path(d, s, by_brand)

            self.topics = self.search_topics(self.root_node)
            #print("num_topics: ", len(self.topics))
            #if len(self.topics) > 50:  # control the total topic number if needed
            #    fix_tree = True
            #else:
            #    fix_tree = False

            # update last weights
            for topic in self.topics:
                topic.word_last_weights = topic.word_cur_weights.copy()

            # - sample levels (topics)
            #logging.info('sample levels')
            for d in range(len(self.corpus)):
                for s in range(len(self.corpus[d])):
                    self.sample_topics(d, s)             # original nCRP
            w_likelihood = self.word_likelihood()

            # optimization M-step
            #logging.info('optimize rating likelihood')
            rating_MSE = self.optimize_likelihood()
            #print(rating_MSE, w_likelihood)

            # update current weights
            self.update_cur_weights()

            # display results every # iterations
            #if (iteration > 0) and ((iteration+1) % display_topics == 0):
            #    print(f" {iteration+1}")
            #    self.print_nodes(n_words, with_weights)
                
                
    def sample_path(self, d, s, by_brand):
        '''
        sample path for each doc with a changing tree structure 
        '''
        brand = self.brands[d]

        # define a path starting from the leaf node of this doc
        path = np.zeros(self.num_levels, dtype=np.object)
        node = self.document_leaves[d][s]
        # e.g. [3, 2, 1, 0] for num_levels = 4
        for level in range(self.num_levels-1, -1, -1):
            path[level] = node
            node = node.parent

        # remove this document from the path, deleting empty nodes if necessary
        self.document_leaves[d][s].drop_path(brand)

        ############################################################
        # calculates the prior p(c_{d,s} | c^{-(d,s)},b) in eq. (7)
        ############################################################

        node_weights = {}
        if by_brand:
            self.calculate_ncrp_prior_by_brand(
                node_weights, self.root_node, 0.0, brand)
        else:
            self.calculate_ncrp_prior(node_weights, self.root_node, 0.0)

        ############################################################
        # calculates the likelihood p(w_{d,s} | c, l) and p(y_d | c, l, beta) in eq. (7)
        ############################################################

        level_word_counts = {}
        for level in range(self.num_levels):
            level_word_counts[level] = {}
        doc_levels = self.levels[d][s]
        doc = self.corpus[d][s]

        # remove doc from path
        for n in range(len(doc)):  # for each word in the doc

            # count the word at each level
            level = doc_levels[n]
            w = doc[n]
            if w not in level_word_counts[level]:
                level_word_counts[level][w] = 1
            else:
                level_word_counts[level][w] += 1

            # remove word count from the node at that level
            level_node = path[level]
            self.hpu_reduce_word_counts(level_node, w, 1)
            level_node.z_d[d] -= 1
            self.w_d[d] -= level_node.omega[brand]
            iter_node = level_node.parent
            while(iter_node):
                iter_node.z_d[d] -= 1
                self.w_d[d] -= iter_node.omega[brand]
                iter_node = iter_node.parent
            level_node.brand_word_counts[brand][w] -= 1

        self.calculate_doc_likelihood(
            d, node_weights, level_word_counts, self.w_d[d])

        ############################################################
        # pick a new path according to the probability above
        ############################################################

        nodes = np.array(list(node_weights.keys()))
        weights = np.array([node_weights[node] for node in nodes])
        # normalise so the largest weight is 1
        weights = np.exp(weights - np.max(weights))
        weights = weights / np.sum(weights)

        choice = self.random_state.multinomial(1, weights).argmax()
        node = nodes[choice]

        # if we picked an internal node, we need to add a new path to the leaf
        if not node.is_leaf():
            node = node.get_new_leaf()

        # add the doc back to the path
        node.add_path(brand)                     # add a customer to the path
        # store the leaf node for this doc
        self.document_leaves[d][s] = node

        # add the words
        # e.g. [2, 1, 0] for num_levels = 3
        for level in range(self.num_levels-1, -1, -1):
            word_counts = level_word_counts[level]
            for w in word_counts:
                self.hpu_add_word_counts(node, w, word_counts[w])
                node.z_d[d] += word_counts[w]
                self.w_d[d] += word_counts[w]*node.omega[brand]
                iter_node = node.parent
                while(iter_node):
                    iter_node.z_d[d] += word_counts[w]
                    self.w_d[d] += word_counts[w] * iter_node.omega[brand]
                    iter_node = iter_node.parent
                node.brand_word_counts[brand][w] += word_counts[w]
            node = node.parent


    def sample_existing_path(self, d, s, by_brand):
        '''
        sample an existing path in current tree for each doc 
        '''

        brand = self.brands[d]

        # define a path starting from the leaf node of this doc
        path = np.zeros(self.num_levels, dtype=np.object)
        node = self.document_leaves[d][s]
        # e.g. [3, 2, 1, 0] for num_levels = 4
        for level in range(self.num_levels-1, -1, -1):
            path[level] = node
            node = node.parent

        # remove this document from the path, deleting empty nodes if necessary
        self.document_leaves[d][s].drop_path(brand)

        ############################################################
        # calculates the prior p(c_{d,s} | c^{-(d,s)},b) in eq. (7) 
        # only for the existing paths (or leaf nodes)
        ############################################################

        total_topic_nodes = self.search_topics(self.root_node)

        leaf_weights = {}
        if by_brand:
            for node in total_topic_nodes:
                if node.is_leaf() and node.brand_customers[brand] > 0:
                    leaf_weights[node] = log(
                        float(node.brand_customers[brand])/self.root_node.brand_customers[brand])
        else:
            for node in total_topic_nodes:
                if node.is_leaf() and node.customers > 0:
                    leaf_weights[node] = log(
                        float(node.customers)/self.root_node.customers)

        ############################################################
        # calculates the likelihood p(w_{d,s} | c, l) and p(y_d | c, l, beta) in eq. (7)
        # only for the existing paths (or leaf nodes)
        ############################################################

        level_word_counts = {}
        for level in range(self.num_levels):
            level_word_counts[level] = {}
        doc_levels = self.levels[d][s]
        doc = self.corpus[d][s]
        # remove doc from path
        for n in range(len(doc)):  # for each word in the doc

            # count the word at each level
            level = doc_levels[n]
            w = doc[n]
            if w not in level_word_counts[level]:
                level_word_counts[level][w] = 1
            else:
                level_word_counts[level][w] += 1

            # remove word count from the node at that level
            level_node = path[level]
            self.hpu_reduce_word_counts(level_node, w, 1)
            level_node.z_d[d] -= 1
            self.w_d[d] -= level_node.omega[brand]
            iter_node = level_node.parent
            while(iter_node):
                iter_node.z_d[d] -= 1
                self.w_d[d] -= iter_node.omega[brand]
                iter_node = iter_node.parent
            level_node.brand_word_counts[brand][w] -= 1

        # likelihood of w_d and y_d
        self.calculate_likelihood_onlyleaf(
            d, leaf_weights, self.root_node, 0.0, level_word_counts, 0, self.w_d[d])

        ############################################################
        # pick a new path according to the probability above
        ############################################################

        nodes = np.array(list(leaf_weights.keys()))
        weights = np.array([leaf_weights[node] for node in nodes])
        # normalise so the largest weight is 1
        weights = np.exp(weights - np.max(weights))
        weights = weights / np.sum(weights)

        choice = self.random_state.multinomial(1, weights).argmax()
        node = nodes[choice]

        # add the doc back to the path
        node.add_path(brand)                     # add a customer to the path
        # store the leaf node for this doc
        self.document_leaves[d][s] = node

        # add the words
        # e.g. [2, 1, 0] for num_levels = 3
        for level in range(self.num_levels-1, -1, -1):
            word_counts = level_word_counts[level]
            for w in word_counts:
                self.w_d[d] += word_counts[w]*node.omega[brand]
                self.hpu_add_word_counts(node, w, word_counts[w])
                node.z_d[d] += word_counts[w]
                iter_node = node.parent
                while(iter_node):
                    iter_node.z_d[d] += word_counts[w]
                    self.w_d[d] += word_counts[w] * iter_node.omega[brand]
                    iter_node = iter_node.parent
                node.brand_word_counts[brand][w] += word_counts[w]
            node = node.parent

    def calculate_ncrp_prior_by_brand(self, node_weights, node, weight, brand):
        ''' Calculates the prior on the path according to the nested CRP '''
        N = len(node.children)
        for child in node.children:
            child_weight = log(float(
                child.brand_customers[brand]+1) / (node.brand_customers[brand] + N + self.gamma))
            self.calculate_ncrp_prior_by_brand(
                node_weights, child, weight + child_weight, brand)
        if node.is_leaf():
            node_weights[node] = weight
        else:
            node_weights[node] = weight + \
                log(self.gamma /
                    (node.brand_customers[brand] + N + self.gamma))

    def calculate_ncrp_prior(self, node_weights, node, weight):
        ''' Calculates the prior on the path according to the nested CRP '''

        for child in node.children:
            if child.customers > 0:
                child_weight = log(float(child.customers) /
                                   (node.customers + self.gamma))
                self.calculate_ncrp_prior(
                    node_weights, child, weight + child_weight)
        if node.is_leaf():
            node_weights[node] = weight
        else:
            node_weights[node] = weight + \
                log(self.gamma / (node.customers + self.gamma))

    def calculate_doc_likelihood(self, d, node_weights, level_word_counts, w_0):

        # calculate the weight for a new path at a given level
        new_topic_weights = np.zeros(self.num_levels)

        for level in range(1, self.num_levels):  # skip the root

            word_counts = level_word_counts[level]
            total_count = 0

            for w in word_counts:
                count = word_counts[w]
                new_topic_weights[level] += count * \
                    log((self.eta) / (self.eta_sum))
                '''
                for i in range(count):
                    new_topic_weights[level] +=  log((self.eta + i) / (self.eta_sum + total_count))
                    total_count += 1
                '''
        self.calculate_likelihood(
            d, node_weights, self.root_node, 0.0, level_word_counts, new_topic_weights, 0, w_0)

    def calculate_likelihood(self, d, node_weights, node, parent_weight, level_word_counts, new_topic_weights, level, w_0):

        # first calculate the likelihood of the words at this level, given this topic
        node_weight = parent_weight
        word_counts = level_word_counts[level]
        total_count = 0

        for w in word_counts:
            count = word_counts[w]
            for i in range(count):
                node_weight += log((self.eta + node.word_weighted_counts[w] + i) /
                                   (self.eta_sum + node.total_weighted_counts + total_count))
                total_count += 1

        brand = self.brands[d]
        #w_sum = w_0 + total_count*node.omega[brand]
        omega_sum = node.omega[brand]
        iter_node = node.parent
        while (iter_node):
            omega_sum += iter_node.omega[brand]
            iter_node = iter_node.parent
        w_sum = w_0 + total_count * omega_sum

        # propagate that weight to the child nodes
        for child in node.children:
            self.calculate_likelihood(d, node_weights, child, node_weight,
                                      level_word_counts, new_topic_weights, level+1, w_sum)

        # finally if this is an internal node, add the weight of a new path
        level += 1
        while level < self.num_levels:
            node_weight += new_topic_weights[level]
            level += 1

        if node in node_weights.keys():
            node_weights[node] += node_weight  # word likelihood
            w_sum = w_sum/(self.document_length[d])
            node_weights[node] += (-(self.ratings[d] - w_sum)
                                   ** 2 / (2 * self.rho_square))

    def calculate_likelihood_onlyleaf(self, d, leaf_weights, node, parent_weight, level_word_counts, level, w_0):

        # calculate the likelihood of the words at this level, given this topic
        node_weight = parent_weight
        word_counts = level_word_counts[level]

        total_count = 0
        for w in word_counts:
            count = word_counts[w]
            for i in range(count):
                node_weight += log((self.eta + node.word_weighted_counts[w] + i) /
                                   (self.eta_sum + node.total_weighted_counts + total_count))
                total_count += 1

        brand = self.brands[d]
        #w_sum = w_0 + total_count*node.omega[brand]
        omega_sum = node.omega[brand]
        iter_node = node.parent
        while (iter_node):
            omega_sum += iter_node.omega[brand]
            iter_node = iter_node.parent
        w_sum = w_0 + total_count * omega_sum

        # propagate that weight to the child nodes
        for child in node.children:
            self.calculate_likelihood_onlyleaf(d, leaf_weights, child, node_weight,
                                             level_word_counts, level+1, w_sum)
        if node.is_leaf() and node in leaf_weights.keys():
            leaf_weights[node] += node_weight  # word likelihood
            w_sum = w_sum/(self.document_length[d])
            # rating likelihood
            leaf_weights[node] += (-(self.ratings[d] - w_sum)
                                   ** 2 / (2 * self.rho_square))

    def sample_topics(self, d, s):
        '''
        sample topics (or levels) along a selected path for each doc
        '''

        doc = self.corpus[d][s]
        brand = self.brands[d]

        # initialise level counts
        doc_levels = self.levels[d][s]
        level_counts = np.zeros(self.num_levels, dtype=np.int)
        for c in doc_levels:
            level_counts[c] += 1

        # get the leaf node and populate the path
        path = np.zeros(self.num_levels, dtype=np.object)
        node = self.document_leaves[d][s]
        # e.g. [3, 2, 1, 0] for num_levels = 4
        for level in range(self.num_levels-1, -1, -1):
            path[level] = node
            node = node.parent

        # sample a new level for each word
        level_weights = np.zeros(self.num_levels)
        for n in range(len(doc)):

            w = doc[n]
            word_level = doc_levels[n]

            # remove from model
            level_counts[word_level] -= 1
            node = path[word_level]
            node.z_d[d] -= 1
            self.w_d[d] -= node.omega[brand]
            iter_node = node.parent
            while(iter_node):
                iter_node.z_d[d] -= 1
                self.w_d[d] -= iter_node.omega[brand]
                iter_node = iter_node.parent
            self.hpu_reduce_word_counts(node, w, 1)
            node.brand_word_counts[brand][w] -= 1

            # pick new level
            for level in range(self.num_levels):
                w_plus = 0
                for i in range(level+1):
                    w_plus += path[i].omega[brand]
                w_1 = (self.w_d[d] + w_plus)/self.document_length[d]
                p_rating = exp(-(self.ratings[d] - w_1) ** 2 / (2 * self.rho_square))
                # sampling weights of each level according to Eq.(11)
                level_weights[level] = (self.alpha + level_counts[level]) * p_rating *                    \
                    (self.eta + path[level].word_weighted_counts[w]) / \
                    (self.eta_sum + path[level].total_weighted_counts)
            level_weights = level_weights / np.sum(level_weights)
            level = self.random_state.multinomial(1, level_weights).argmax()

            # put the word back into the model
            doc_levels[n] = level
            level_counts[level] += 1
            node = path[level]
            self.hpu_add_word_counts(node, w, 1)
            node.brand_word_counts[brand][w] += 1
            node.z_d[d] += 1
            self.w_d[d] += node.omega[brand]
            iter_node = node.parent
            while(iter_node):
                iter_node.z_d[d] += 1
                self.w_d[d] += iter_node.omega[brand]
                iter_node = iter_node.parent
    
    def word_likelihood(self):
        logp = 0
        N = 0
        for d in range(len(self.corpus)):
            for s in range(len(self.corpus[d])):
                N += len(self.corpus[d][s])
                # get the leaf node and populate the path
                path = np.zeros(self.num_levels, dtype=np.object)
                node = self.document_leaves[d][s]
                for level in range(self.num_levels-1, -1, -1): # e.g. [3, 2, 1, 0] for num_levels = 4
                    path[level] = node
                    node = node.parent
                theta = np.ones(self.num_levels)
                for n in range(len(self.corpus[d][s])):
                    word_level = self.levels[d][s][n]
                    theta[word_level] += 1
                theta /= np.sum(theta)
                for n in range(len(self.corpus[d][s])):
                    w = self.corpus[d][s][n]
                    p_w = 0
                    for word_level in range(self.num_levels):
                        node = path[word_level]
                        p_w += theta[word_level] * (self.eta + node.word_counts[w] ) / (self.eta_sum + np.sum(node.word_counts ) )
                    logp += log(p_w)
        return logp/N

    def optimize_likelihood(self):
        self.topics = []
        z_d = []  # topic_num * doc_num
        queue = [self.root_node]
        # search all topics along the tree
        while (len(queue) > 0):
            node = queue.pop(0)
            self.topics.append(node)
            z_d.append(node.z_d)
            for child in node.children:
                queue.append(child)
        z_d = np.array(z_d)
        N_d = self.document_length.reshape((self.num_documents, 1))
        SSE = 0
        for brand in range(self.brand_num):
            brand_idx0, brand_idx1 = self.brand_idx[brand], self.brand_idx[brand+1]
            # ols for each brand
            z_1 = z_d.T[brand_idx0:brand_idx1, :]/N_d[brand_idx0:brand_idx1, :]
            y_1 = self.ratings.reshape((self.num_documents, 1))[
                brand_idx0:brand_idx1, :]
            w_1 = np.linalg.pinv(np.dot(z_1.T, z_1))@z_1.T@y_1
            # update regression coefficients
            for i in range(len(self.topics)):
                node = self.topics[i]
                node.omega[brand] = w_1.reshape((len(self.topics),))[i]
            self.w_d[brand_idx0:brand_idx1] = (
                z_d.T[brand_idx0:brand_idx1, :]@w_1).reshape((brand_idx1-brand_idx0,))
            # compute SSE of rating prediction for each brand
            SSE += np.sum((z_1 @ w_1 - y_1)**2)
        # compute MSE
        MSE = SSE/self.num_documents
        return MSE


    def update_cur_weights(self):
        for topic in self.topics:
            # update polya urn sampling weights
            topic.word_cur_weights = np.ones(len(self.vocab))
            if topic.level < self.num_levels-1:
                c_topics = topic.children
                ent = 0
                net_cumu_phi_sum = sum(
                    [np.sum(c_topic.word_counts) for c_topic in c_topics])
                if net_cumu_phi_sum == 0:
                    continue
                for c_topic in c_topics:
                    p = np.sum(c_topic.word_counts)/net_cumu_phi_sum
                    if p > 0:
                        ent -= p*math.log(p)

                for w in range(len(self.vocab)):
                    ent_w = 0
                    net_cumu_phi_w = sum([c_topic.word_counts[w]
                                         for c_topic in c_topics])
                    if net_cumu_phi_w == 0:
                        continue
                    for c_topic in c_topics:
                        p = c_topic.word_counts[w]/net_cumu_phi_w
                        if p > 0:
                            ent_w -= p*math.log(p)
                    if ent > ent_w:
                        topic.word_cur_weights[w] = round(ent_w/ent, 3)

    def coherence(self, n_words):
        self.topics = self.search_topics(self.root_node)
        topics_top_words0 = []
        topics_top_words1 = []
        topics_top_words2 = []
        for topic in self.topics:
            top_idx = np.argsort(topic.word_counts)[::-1][:n_words] 
            top_words = [self.vocab[idx] for idx in top_idx]
            if topic.level == 0:
                topics_top_words0.append(top_words)
            elif topic.level == 1:
                topics_top_words1.append(top_words)
            elif topic.level == 2 and topic.customers>1:
                topics_top_words2.append(top_words)      
        docs = []
        for d in range(len(self.corpus)):
            for s in range(len(self.corpus[d])):
                doc = self.corpus[d][s]
                doc = [self.vocab[w] for w in doc]
                docs.append(doc)               
        bow_corpus = [self.vocab.doc2bow(doc) for doc in docs]
            
        cm0 = CoherenceModel(topics = topics_top_words0, dictionary = self.vocab, corpus = bow_corpus, coherence='u_mass' )
        score0 = cm0.get_coherence()
        cm1 = CoherenceModel(topics = topics_top_words1, dictionary = self.vocab, corpus = bow_corpus, coherence='u_mass' )
        score1 = cm1.get_coherence()
        cm2 = CoherenceModel(topics = topics_top_words2, dictionary = self.vocab, corpus = bow_corpus, coherence='u_mass' )
        score2 = cm2.get_coherence()
        cm = CoherenceModel(topics = topics_top_words0+topics_top_words1+topics_top_words2, dictionary = self.vocab, corpus = bow_corpus, coherence='u_mass' )
        score = cm.get_coherence()
        return score, score0, score1, score2           

    def print_nodes(self, n_words, with_weights):
        self.print_node(self.root_node, 0, n_words, with_weights)

    def print_node(self, node, indent, n_words, with_weights):
        out = '    ' * indent
        out += 'topic=%d level=%d (documents=%d): ' % (node.node_id,
                                                       node.level, node.customers)
        out += node.get_top_words(n_words, with_weights)
        for brand in range(self.brand_num):
            coef = node.omega[brand]
            iter_node = node.parent
            while(iter_node):
                coef += iter_node.omega[brand]
                iter_node = iter_node.parent
            out += (' '+str(coef))
        print(out)
        for child in node.children:
            if child.customers > 1000:
                self.print_node(child, indent+1, n_words, with_weights)

    def print_regression_coef(self):
        self.print_coef(self.root_node, 0)

    def print_coef(self, node, indent):
        #out = '    ' * indent
        #out += 'topic=%d level=%d (documents=%d): ' % (node.node_id, node.level, node.customers)

        out = 'level=%d ' % (node.level)
        out += ' %.4f %.4f' % (node.diff[0], node.diff[1])

        print(out)
        for child in node.children:
            self.print_coef(child, indent+1)

    def Affinity(self):
        topics1 = []
        topics2 = []
        for topic in self.search_topics(self.root_node):
            if topic.level == 1:
                topics1.append(topic)
            if topic.level == 2 and topic.customers>1:
                topics2.append(topic)
        cosine_child = []
        cosine_nonchild = []
        for t1 in topics1:
            for t2 in topics2:
                if t2 in t1.children:
                    score = cosine_similarity(t1.word_weighted_counts.reshape(
                        1,-1),t2.word_weighted_counts.reshape(1,-1))[0][0]
                    cosine_child.append(score)
                else:
                    score = cosine_similarity(t1.word_weighted_counts.reshape(
                        1,-1),t2.word_weighted_counts.reshape(1,-1))[0][0]
                    cosine_nonchild.append(score)

        return (np.mean(cosine_child)/np.mean(cosine_nonchild))



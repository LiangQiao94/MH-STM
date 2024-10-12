# -*- coding: utf-8 -*-
import seaborn as sns
import random, bisect
from numpy.random import RandomState
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log

data_dir = "./data/simulation"

class NCRPNode(object):

    # class variable to keep track of total nodes created so far
    total_nodes = 0
    last_node_id = 0

    def __init__(self, num_levels, brand_num, eta, parent=None, level=0,
                 random_state=None):

        self.node_id = NCRPNode.last_node_id
        NCRPNode.last_node_id += 1
        
        self.eta = eta
        self.phi = np.random.dirichlet(eta)
        self.customers = 0
        self.brand_customers = np.zeros(brand_num)
        self.parent = parent
        self.children = []
        self.level = level
        self.num_levels = num_levels
        self.brand_num = brand_num
        # 3d coordinates in the predifined cube
        self.identifier = None
        # generate regression parameters for each brand
        self.beta = np.random.normal(0, self.level+1, self.brand_num)

        if random_state is None:
            self.random_state = RandomState()
        else:
            self.random_state = random_state


    def add_child(self):
        ''' Adds a child to the next level of this node '''
        node = NCRPNode(self.num_levels, self.brand_num, self.eta, parent=self, level=self.level+1)
        self.children.append(node)
        NCRPNode.total_nodes += 1
        return node

    def is_leaf(self):
        ''' Check if this node is a leaf node '''
        return self.level == self.num_levels-1


    def remove(self, node):
        ''' Removes a child node '''
        self.children.remove(node)
        NCRPNode.total_nodes -= 1



if __name__ == "__main__":

    
    # random seed for generating training data
    random.seed(10)
    np.random.seed(9)
    '''
    # random seed for generating test data
    random.seed(1)
    np.random.seed(0)    
    '''
    
    # predefined parameters
    N_s = 5 # averge num of sentences in a review
    N_w = 10 # averge num of words in a sentence
    rho = 1.0
    num_levels = 3
    D = 200 # num of reviews under each brand in corpus
    branches = [3,2,4]
    num_paths = sum(branches)
    V = 100
    num_brands = 10
    alpha = [1] * num_levels
    eta = 0.01
    
    
    # generate 3-level topic tree structure
    root_node = NCRPNode(num_levels, num_brands, [eta]*V)
    root_node.identifier = "1"
    leaves = []
    for i in range(len(branches)):
        node = root_node.add_child()
        node.identifier = root_node.identifier + '-' + str(i+1)
        for j in range(branches[i]):
            child = node.add_child()
            child.identifier = node.identifier + '-' + str(j+1)
            leaves.append(child)
            node.phi += child.phi
        node.phi /= np.sum(node.phi)
        root_node.phi += node.phi
    root_node.phi /= np.sum(root_node.phi)
    
    path = np.zeros(len(leaves), dtype=np.object)
    for i in range(len(leaves)):
        path[i] = np.zeros(num_levels, dtype=np.object)
        node = leaves[i]
        for level in range(num_levels-1, -1, -1): # e.g. [2, 1, 0] for num_levels = 3
            path[i][level] = node
            node = node.parent
    
    
    # generate brand distributions over tree
    brand_distribution = np.zeros(num_brands, dtype=np.object)
    for b in range(num_brands):
         random_weights = np.random.randint(1, 100, size = num_paths)
         brand_distribution[b] = random_weights/np.sum(random_weights)

    
    # generate simulated documents
    write_file = open(data_dir + 'simulated_corpus_'+ '-'.join([str(ele) for ele in branches]+[str(eta)]) + '.txt', 'w')   
    leaf_label = open(data_dir + 'leaf_label_'+ '-'.join([str(ele) for ele in branches]+[str(eta)]) + '.txt', 'w') 
    LL = 0
    N_total = 0
    for b in range(num_brands):
        for i in range(D):
            sentence_num = np.random.poisson(lam=N_s)
            while(sentence_num == 0):
                sentence_num = np.random.poisson(lam = N_s)
            doc = ""
            overall = 0
            N = 0            
            for j in range(sentence_num):
                sentence = ""
                words_num = np.random.poisson(lam = N_w)
                while (words_num == 0):
                    words_num = np.random.poisson(lam = N_w)
                N += words_num
                N_total += words_num
                theta = np.random.dirichlet(alpha)
                # select path for each sentence
                leaf_choice = np.random.multinomial(1, brand_distribution[b]).argmax()
                leaf_node = leaves[leaf_choice]
                leaf_label.write(str(leaf_choice) +"\n")
                # select topics along the path
                for n in range(words_num):
                    level = np.random.multinomial(1, theta).argmax()
                    select_topic = path[leaf_choice][level]
                    # add to overall score
                    for l in range(level,-1,-1):
                        overall += path[leaf_choice][l].beta[b]
                    # generate a specific word under the topic
                    w = np.random.multinomial(1, select_topic.phi).argmax()
                    LL += log(select_topic.phi[w])
                    sentence += str(w)
                    sentence += " "
                sentence += "."
                doc += sentence
            overall /= N
            overall = np.random.normal(overall,rho)
            write_file.write(doc + "\t" + str(overall) + "\t" + str(b) +"\n")   
    write_file.close()
    leaf_label.close()
    print(LL/N_total)
    
    aspect_brand_ranking = open(data_dir + 'brand_ranking_'+ '-'.join([str(ele) for ele in branches]+[str(eta)]) + '.txt', 'w')
    for i in range(num_paths):
        for b in range(num_brands):
            score = leaves[i].beta[b] + leaves[i].parent.beta[b] + leaves[i].parent.parent.beta[b]         
            aspect_brand_ranking.write(str(score) +'\t')
        aspect_brand_ranking.write('\n')
    aspect_brand_ranking.close()
    
    

        
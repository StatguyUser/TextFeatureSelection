#!/usr/bin/env python
# coding: utf-8
"""Text features selection."""

# Author: Md Azimul Haque <github.com/StatguyUser>
# License: BSD 3 clause

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score,auc,precision_score,recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
from collections import Counter
import random as rd
import time
import gc
import warnings

warnings.filterwarnings('ignore')
rd.seed(1)


class TextFeatureSelection():
    """Compute score for each word to identify and select words which result in better model performance.

    Parameters
    ----------
    target : list object which has categories of labels. for more than one category, no need to dummy code and instead provide label encoded values as list object.

    input_doc_list : List object which has text. each element of list is text corpus. No need to tokenize, as text will be tokenized in the module while processing. target and input_doc_list should have same length.

    stop_words : Words for which you will not want to have metric values calculated. Default is blank.

    metric_list : List object which has the metric to be calculated. There are 4 metric which are being computed as 'MI','CHI','PD','IG'. you can specify one or more than one as a list object. Default is ['MI','CHI','PD','IG'].    

    Returns
    -------
    values_df : pandas dataframe with results. unique words and score from the desried metric.

    Examples
    --------
    The following example shows how to retrieve the 5 most informative
    features in the Friedman #1 dataset.

    >>> from sklearn.feature_selection.text import TextFeatureSelection

    >>> #Multiclass classification problem
    >>> input_doc_list=['i am very happy','i just had an awesome weekend','this is a very difficult terrain to trek. i wish i stayed back at home.','i just had lunch','Do you want chips?']
    >>> target=['Positive','Positive','Negative','Neutral','Neutral']
    >>> result_df=TextFeatureSelection(target=target,input_doc_list=input_doc_list).getScore()
    >>> print(result_df)

        word list  word occurence count  Proportional Difference  Mutual Information  Chi Square  Information Gain
    0          am                     1                      1.0            0.916291    1.875000          0.089257
    1          an                     1                      1.0            0.916291    1.875000          0.089257
    2          at                     1                      1.0            1.609438    5.000000          0.000000
    3     awesome                     1                      1.0            0.916291    1.875000          0.089257
    4        back                     1                      1.0            1.609438    5.000000          0.000000
    5       chips                     1                      1.0            0.916291    1.875000          0.089257
    6   difficult                     1                      1.0            1.609438    5.000000          0.000000
    7          do                     1                      1.0            0.916291    1.875000          0.089257
    8         had                     2                      1.0            0.223144    0.833333          0.008164
    9       happy                     1                      1.0            0.916291    1.875000          0.089257
    10       home                     1                      1.0            1.609438    5.000000          0.000000
    11         is                     1                      1.0            1.609438    5.000000          0.000000
    12       just                     2                      1.0            0.223144    0.833333          0.008164
    13      lunch                     1                      1.0            0.916291    1.875000          0.089257
    14     stayed                     1                      1.0            1.609438    5.000000          0.000000
    15    terrain                     1                      1.0            1.609438    5.000000          0.000000
    16       this                     1                      1.0            1.609438    5.000000          0.000000
    17         to                     1                      1.0            1.609438    5.000000          0.000000
    18       trek                     1                      1.0            1.609438    5.000000          0.000000
    19       very                     2                      1.0            0.916291    2.222222          0.008164
    20       want                     1                      1.0            0.916291    1.875000          0.089257
    21    weekend                     1                      1.0            0.916291    1.875000          0.089257
    22       wish                     1                      1.0            1.609438    5.000000          0.000000
    23        you                     1                      1.0            0.916291    1.875000          0.089257



    >>> #Binary classification
    >>> input_doc_list=['i am content with this location','i am having the time of my life','you cannot learn machine learning without linear algebra','i want to go to mars']
    >>> target=[1,1,0,1]
    >>> result_df=TextFeatureSelection(target=target,input_doc_list=input_doc_list).getScore()
    >>> print(result_df)
       word list  word occurence count  Proportional Difference  Mutual Information  Chi Square  Information Gain
    0    algebra                     1                     -1.0            1.386294    4.000000               0.0
    1         am                     2                      1.0                -inf    1.333333               0.0
    2     cannot                     1                     -1.0            1.386294    4.000000               0.0
    3    content                     1                      1.0                -inf    0.444444               0.0
    4         go                     1                      1.0                -inf    0.444444               0.0
    5     having                     1                      1.0                -inf    0.444444               0.0
    6      learn                     1                     -1.0            1.386294    4.000000               0.0
    7   learning                     1                     -1.0            1.386294    4.000000               0.0
    8       life                     1                      1.0                -inf    0.444444               0.0
    9     linear                     1                     -1.0            1.386294    4.000000               0.0
    10  location                     1                      1.0                -inf    0.444444               0.0
    11   machine                     1                     -1.0            1.386294    4.000000               0.0
    12      mars                     1                      1.0                -inf    0.444444               0.0
    13        my                     1                      1.0                -inf    0.444444               0.0
    14        of                     1                      1.0                -inf    0.444444               0.0
    15       the                     1                      1.0                -inf    0.444444               0.0
    16      this                     1                      1.0                -inf    0.444444               0.0
    17      time                     1                      1.0                -inf    0.444444               0.0
    18        to                     1                      1.0                -inf    0.444444               0.0
    19      want                     1                      1.0                -inf    0.444444               0.0
    20      with                     1                      1.0                -inf    0.444444               0.0
    21   without                     1                     -1.0            1.386294    4.000000               0.0
    22       you                     1                     -1.0            1.386294    4.000000               0.0


    Notes
    -----
    Chi-square (CHI):
     - It measures the lack of independence between t and c.
     - It has a natural value of zero if t and c are independent. If it is higher, then term is dependent
     - It is not reliable for low-frequency terms
     - For multi-class categories, we will calculate X^2 value for all categories and will take the Max(X^2) value across all categories at the word level.
     - It is not to be confused with chi-square test and the values returned are not significance values

    Mutual information (MI):
     - Rare terms will have a higher score than common terms.
     - For multi-class categories, we will calculate MI value for all categories and will take the Max(MI) value across all categories at the word level.

    Proportional difference (PD):
     - How close two numbers are from becoming equal. 
     - Helps ﬁnd unigrams that occur mostly in one class of documents or the other
     - We use the positive document frequency and negative document frequency of a unigram as the two numbers.
     - If a unigram occurs predominantly in positive documents or predominantly in negative documents then the PD will be close to 1, however if distribution of unigram is almost similar, then PD is close to 0.
     - We can set a threshold to decide which words to be included
     - For multi-class categories, we will calculate PD value for all categories and will take the Max(PD) value across all categories at the word level.

Information gain (IG):
     - It gives discriminatory power of the word

    References
    ----------
    Yiming Yang and Jan O. Pedersen "A Comparative Study on Feature Selection in Text Categorization"
    http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E5CC43FE63A1627AB4C0DBD2061FE4B9?doi=10.1.1.32.9956&rep=rep1&type=pdf

    Christine Largeron, Christophe Moulin, Mathias Géry "Entropy based feature selection for text categorization"
    https://hal.archives-ouvertes.fr/hal-00617969/document

    Mondelle Simeon, Robert J. Hilderman "Categorical Proportional Difference: A Feature Selection Method for Text Categorization"
    https://pdfs.semanticscholar.org/6569/9f0e1159a40042cc766139f3dfac2a3860bb.pdf
    
    Tim O`Keefe and Irena Koprinska "Feature Selection and Weighting Methods in Sentiment Analysis"
    https://www.researchgate.net/publication/242088860_Feature_Selection_and_Weighting_Methods_in_Sentiment_Analysis
    """
    
    def __init__(self,target,input_doc_list,stop_words=None,metric_list=['MI','CHI','PD','IG']):
        self.target=target
        self.input_doc_list=input_doc_list
        self.stop_words=stop_words
        self.metric_list=metric_list
 
    def _ChiSquare(self,A,B,C,D,N):
        with np.errstate(divide='ignore', invalid='ignore'):
            return (N*((A*D)-(C*B))**2)/((A+B)*(A+C)*(B+D)*(C+D))

    def _MutualInformation(self,A,B,C,N):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log((A*N)/((A+C)*(A+B)))
    
    def _InformationGain(self,A,B,C,D,N):
        with np.errstate(divide='ignore', invalid='ignore'):
            return (-((A+C)/N)*np.log((A+C)/N))+(A/N)*np.log(A/(A+B))+(C/N)*np.log(C/(C+D))

    def _ProportionalDifference(self,A,B):
        with np.errstate(divide='ignore', invalid='ignore'):
            return ((A-B)*(-1))/(A+B)
    
    def _get_binary_label(self,label_array):
        #get numpy array
        label_array=np.array(label_array)
        unique_label=np.unique(label_array)
        #if not binary coded already, do so
        if 0 in unique_label and 1 in unique_label:
            pass
        else:
            label_array=np.where(label_array==unique_label[0],1,0)
        return label_array

    def _get_term_binary_matrix(self,input_doc_list):

        #initialize vectorizer
        if self.stop_words:
            #unique word and word count
            vectorizer = CountVectorizer(stop_words=self.stop_words)
            X = vectorizer.fit_transform(input_doc_list)
            word_list = vectorizer.get_feature_names()

            #binary word document matrix
            vectorizer = CountVectorizer(binary=True,stop_words=self.stop_words)
            X = vectorizer.fit_transform(input_doc_list)
            word_binary_matrix = X.toarray()
            count_list = word_binary_matrix.sum(axis=0)
            
            ##return
            return word_list,count_list,word_binary_matrix
        else:
            #unique word and word count
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(input_doc_list)
            word_list = vectorizer.get_feature_names()

            #binary word document matrix
            vectorizer = CountVectorizer(binary=True)
            X = vectorizer.fit_transform(input_doc_list)
            word_binary_matrix = X.toarray()
            count_list = word_binary_matrix.sum(axis=0)
            ##return
            return word_list,count_list,word_binary_matrix

    def _get_ABCD(self,word_binary_matrix,label_array):

        A=[]
        B=[]
        C=[]
        D=[]
        for i in range(word_binary_matrix.shape[1]):
            computed_result=Counter(label_array * 2 + word_binary_matrix[:,i])
            A.append(computed_result[1])
            B.append(computed_result[3])
            C.append(computed_result[0])
            D.append(computed_result[2])

        A=np.array(A)
        B=np.array(B)
        C=np.array(C)
        D=np.array(D)
        N=A+B+C+D
        return A,B,C,D,N
    
    
    def _getvalues_singleclass(self):
                
        #get binary labels
        label_array=self._get_binary_label(self.target)

        #get word, count, binary matrix
        word_list,count_list,word_binary_matrix=self._get_term_binary_matrix(self.input_doc_list)

        #get ABCDN
        A,B,C,D,N=self._get_ABCD(word_binary_matrix,label_array)
        
        #create DF
        out_df=pd.DataFrame({'word list':word_list,'word occurence count':count_list})
        if 'PD' in self.metric_list:
            out_df['Proportional Difference']=self._ProportionalDifference(A,B)
        if 'MI' in self.metric_list:
            out_df['Mutual Information']=self._MutualInformation(A,B,C,N)
        if 'CHI' in self.metric_list:
            out_df['Chi Square']=self._ChiSquare(A,B,C,D,N)
        if 'IG' in self.metric_list:
            out_df['Information Gain']=self._InformationGain(A,B,C,D,N)
            out_df['Information Gain'].replace(np.nan,0,inplace=True)

        return out_df
    
    def _getvalues_multiclass(self):
        
        #labels as numpy array
        numpy_target=np.array(self.target)
        
        #get word, count, binary matrix
        word_list,count_list,word_binary_matrix=self._get_term_binary_matrix(self.input_doc_list)
        result_dict={}

        #for each class
        for calc_base_label in list(set(self.target)):
            #get binary labels
            label_array=np.where(numpy_target==calc_base_label,1,0)

            #get ABCDN
            B,A,D,C,N=self._get_ABCD(word_binary_matrix,label_array)
            
            #create DF
            out_df=pd.DataFrame({'word list':word_list,'word occurence count':count_list})

            if 'PD' in self.metric_list:
                out_df['Proportional Difference']=self._ProportionalDifference(A,B)
            if 'MI' in self.metric_list:
                out_df['Mutual Information']=self._MutualInformation(A,B,C,N)
            if 'CHI' in self.metric_list:
                out_df['Chi Square']=self._ChiSquare(A,B,C,D,N)
            if 'IG' in self.metric_list:
                out_df['Information Gain']=self._InformationGain(A,B,C,D,N)
                out_df['Information Gain'].replace(np.nan,0,inplace=True)

            ##assign to dict for master calculation
            result_dict[calc_base_label]=out_df
        
        ####merge
        final_results_pd=pd.DataFrame()
        final_results_mi=pd.DataFrame()
        final_results_chi=pd.DataFrame()
        final_results_ig=pd.DataFrame()
        
        #final result
        final_results=pd.DataFrame({'word list':out_df['word list'],'word occurence count':out_df['word occurence count']})
        
        for calc_base_label in list(set(self.target)):
            if 'PD' in self.metric_list:
                label_df_pd=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'PD_'+str(calc_base_label):result_dict[calc_base_label]['Proportional Difference']})
                if final_results_pd.shape[0]:
                    final_results_pd=final_results_pd.merge(label_df_pd,on=['word list'])
                else:
                    final_results_pd=label_df_pd

                ##final calculation
                if calc_base_label==list(set(self.target))[-1]:
                    label_df_pd=pd.DataFrame({'word list':final_results_pd['word list'],
                                              'Proportional Difference':final_results_pd.max(axis=1)})
                    #assign to final result df
                    if final_results.shape[0]:
                        final_results=final_results.merge(label_df_pd,on=['word list'])
                    else:
                        final_results=label_df_pd

            
            if 'MI' in self.metric_list:
                label_df_mi=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'MI_'+str(calc_base_label):result_dict[calc_base_label]['Mutual Information']})
                if final_results_mi.shape[0]:
                    final_results_mi=final_results_mi.merge(label_df_mi,on=['word list'])
                else:
                    final_results_mi=label_df_mi

                ##final calculation
                if calc_base_label==list(set(self.target))[-1]:
                    label_df_mi=pd.DataFrame({'word list':final_results_mi['word list'],
                                              'Mutual Information':final_results_mi.max(axis=1)})
                    #assign to final result df
                    if final_results.shape[0]:
                        final_results=final_results.merge(label_df_mi,on=['word list'])
                    else:
                        final_results=label_df_mi
                    

            if 'CHI' in self.metric_list:
                label_df_chi=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'CHI_'+str(calc_base_label):result_dict[calc_base_label]['Chi Square']})
                if final_results_chi.shape[0]:
                    final_results_chi=final_results_chi.merge(label_df_chi,on=['word list'])
                else:
                    final_results_chi=label_df_chi

                ##final calculation
                if calc_base_label==list(set(self.target))[-1]:
                    label_df_chi=pd.DataFrame({'word list':final_results_chi['word list'],
                                              'Chi Square':final_results_chi.max(axis=1)})
                    #assign to final result df
                    if final_results.shape[0]:
                        final_results=final_results.merge(label_df_chi,on=['word list'])
                    else:
                        final_results=label_df_chi

            if 'IG' in self.metric_list:
                label_df_ig=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'IG_'+str(calc_base_label):result_dict[calc_base_label]['Information Gain']})
                if final_results_ig.shape[0]:
                    final_results_ig=final_results_ig.merge(label_df_ig,on=['word list'])
                else:
                    final_results_ig=label_df_ig
                    
                ##final calculation
                if calc_base_label==list(set(self.target))[-1]:
                    label_df_ig=pd.DataFrame({'word list':final_results_ig['word list'],
                                              'Information Gain':final_results_ig.max(axis=1)})
                    #assign to final result df
                    if final_results.shape[0]:
                        final_results=final_results.merge(label_df_ig,on=['word list'])
                    else:
                        final_results=label_df_ig
        
        return final_results

    
    def getScore(self):
        if type(self.target)==list and type(self.input_doc_list)==list:            
            if len(self.target)!=len(self.input_doc_list):
                print('Please provide target and input_doc_list of similar length.')
            else:
                if len(set(self.target))==2:
                    values_df=self._getvalues_singleclass()
                    return values_df
                elif len(set(self.target))>2:
                    values_df=self._getvalues_multiclass()
                    return values_df
        else:
            print('Please provide target and input_doc_list both as list object.')

            


class TextFeatureSelectionGA():
    '''Use genetic algorithm for selecting text tokens which give best classification results
    
    Genetic Algorithm Parameters
    ----------
    
    generations : Number of generations to run genetic algorithm. 500 as deafult, as used in the original paper
    
    population : Number of individual chromosomes. 50 as default, as used in the original paper
    
    prob_crossover : Probability of crossover. 0.9 as default, as used in the original paper
    
    prob_mutation : Probability of mutation. 0.1 as default, as used in the original paper
    
    percentage_of_token : Percentage of word features to be included in a given chromosome.
        50 as default, as used in the original paper.

    runtime_minutes : Number of minutes to run the algorithm. This is checked in between generations.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from generations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.
        
    References
    ----------
    Noria Bidi and Zakaria Elberrichi "Feature Selection For Text Classification Using Genetic Algorithms"
    https://ieeexplore.ieee.org/document/7804223
    
    '''
    
    def __init__(self,generations=500,population=50,prob_crossover=0.9,prob_mutation=0.1,percentage_of_token=50,runtime_minutes=120):
        self.generations=generations
        self.population=population
        self.prob_crossover=prob_crossover
        self.prob_mutation=prob_mutation
        self.percentage_of_token=percentage_of_token
        self.runtime_minutes=runtime_minutes
        
    def _cost_function_value(self,y_test,y_test_pred,cost_function,avrg):
        if cost_function == 'f1':
            if avrg == 'micro':
                metric=f1_score(y_test,y_test_pred,average='micro')
            if avrg == 'macro':
                metric=f1_score(y_test,y_test_pred,average='macro')
            if avrg == 'samples':
                metric=f1_score(y_test,y_test_pred,average='samples')
            if avrg == 'weighted':
                metric=f1_score(y_test,y_test_pred,average='weighted')
            if avrg == 'binary':
                metric=f1_score(y_test,y_test_pred,average='binary')

        elif cost_function == 'precision':
            if avrg == 'micro':
                metric=precision_score(y_test,y_test_pred,average='micro')
            if avrg == 'macro':
                metric=precision_score(y_test,y_test_pred,average='macro')
            if avrg == 'samples':
                metric=precision_score(y_test,y_test_pred,average='samples')
            if avrg == 'weighted':
                metric=precision_score(y_test,y_test_pred,average='weighted')
            if avrg == 'binary':
                metric=precision_score(y_test,y_test_pred,average='binary')

        elif cost_function == 'recall':
            if avrg == 'micro':
                metric=recall_score(y_test,y_test_pred,average='micro')
            if avrg == 'macro':
                metric=recall_score(y_test,y_test_pred,average='macro')
            if avrg == 'samples':
                metric=recall_score(y_test,y_test_pred,average='samples')
            if avrg == 'weighted':
                metric=recall_score(y_test,y_test_pred,average='weighted')
            if avrg == 'binary':
                metric=recall_score(y_test,y_test_pred,average='binary')

        return metric


    def _computeFitness(self,gene,unique_words,x,y,model,model_metric,avrg,analyzer,min_df,max_df,stop_words,tokenizer,token_pattern,lowercase):
        ### create tfidf matrix for only terms which are in gene
        # get terms from gene and vocabulary combnation
        term_to_use=list(np.array(unique_words)[list(map(bool,gene))])

        metric_result=[]
        skfold=StratifiedKFold(n_splits=5)

        ##get words based on gene index to get vocabulary
        term_to_use=list(np.array(unique_words)[list(map(bool,gene))])

        for train_index, test_index in skfold.split(x,y):
            #get x_train,y_train  x_test,y_test
            X_train, X_test = list(np.array(x)[train_index]),list(np.array(x)[test_index]) 
            y_train, y_test = np.array(y)[train_index],np.array(y)[test_index]

            ##based on vocabulary set, create tfidf matrix for train and test data
            tfidf=TfidfVectorizer(vocabulary=term_to_use,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
            tfidfvec_vectorizer=tfidf.fit(X_train)

            #get x train and test
            X_train=tfidfvec_vectorizer.transform(X_train)
            X_test=tfidfvec_vectorizer.transform(X_test)

            #train model
            model.fit(X_train,y_train)

            #predict probability for test
            y_test_pred=model.predict(X_test)

            #get desired metric and append to metric_result
            metric_result.append(self._cost_function_value(y_test,y_test_pred,model_metric,avrg))

        return np.mean(metric_result)



    def _check_unmatchedrows(self,population_matrix,population_array):
        pop_check=0
        #in each row of population matrix
        for pop_so_far in range(population_matrix.shape[0]):
            #check if it is duplicate
            if sum(population_matrix[pop_so_far]!=population_array)==population_array.shape[0]:
                #assign 1 as value if it is duplicate and break loop
                pop_check=1
                break

        return pop_check

    def _get_population(self,population,population_matrix,population_array):
        iterate=0
        ##append until population and no duplicate chromosome
        while population_matrix.shape[0] < population:
            ##prepare population matrix
            rd.shuffle(population_array)
            #check if it is first iteration, if yes append
            if iterate==0:
                population_matrix = np.vstack((population_matrix,population_array))
                iterate+=1
            #if second iteration and one chromosome already, check if it is duplicate
            elif iterate==1 and sum(population_matrix[0]==population_array)!=population_array.shape[0]:
                population_matrix = np.vstack((population_matrix,population_array))
                iterate+=1
            #when iteration second and beyond check duplicacy
            elif iterate>1 and self._check_unmatchedrows(population_matrix,population_array)==0:
                population_matrix = np.vstack((population_matrix,population_array))

        return population_matrix


    def _get_parents(self,population_array,population_matrix,unique_words,x,y,model,model_metric,avrg,analyzer,min_df,max_df,stop_words,tokenizer,token_pattern,lowercase):

        #keep space for best chromosome
        parents = np.empty((0,population_array.shape[0]))

        #get 6 unique index to fetch from population
        indexes=np.random.randint(0,population_matrix.shape[0],6)
        while len(np.unique(indexes))<6:
            indexes=np.random.randint(0,len(population_matrix),6)

        #mandatory run twice as per GA algorithm
        for run_range in range(2):
            #get 3 unique index to fetch from population
            #if first run then until half
            if run_range==0:
                index_run=indexes[0:3]
            #if second run then from half till end
            else:
                index_run=indexes[3:]

            ##gene pool 1
            gene_1 = population_matrix[index_run[0]]
            #cost of gene 1
            cost1=self._computeFitness(gene=gene_1,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
            ##gene pool 2
            gene_2 = population_matrix[index_run[1]]
            #cost of gene 2
            cost2=self._computeFitness(gene=gene_2,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
            ##gene pool 3
            gene_3 = population_matrix[index_run[2]]
            #cost of gene 3
            cost3=self._computeFitness(gene=gene_3,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)

            #get best chromosome from 3 and assign best chromosome.
            if cost1==max(cost1,cost2,cost3):
                parents = np.vstack((parents,gene_1))
            elif cost2==max(cost1,cost2,cost3):
                parents = np.vstack((parents,gene_2))
            elif cost3==max(cost1,cost2,cost3):
                parents = np.vstack((parents,gene_3))

        #get 2 best chromosome identified as parents
        return parents[0],parents[1]

    def _crossover(self,parent1,parent2,prob_crossover):

        #placeholder for child chromosome
        child1 = np.empty((0,len(parent1)))
        child2 = np.empty((0,len(parent2)))

        #generate random number ofr crossover probability
        crsvr_rand_prob = np.random.rand()

        ## if random decimal generated is less than probability of crossover set
        if crsvr_rand_prob < prob_crossover:
            index1 = np.random.randint(0,len(parent1))
            index2 = np.random.randint(0,len(parent1))

            # get different indices
            # to make sure you will crossover at least one gene
            while index1 == index2:
                index2 = np.random.randint(0,len(parent1))

            index_parent1 = min(index1,index2) 
            index_parent2 = max(index1,index2) 

            ## Parent 1
            # first segment
            first_seg_parent1 = parent1[:index_parent1]
            # middle segment; where the crossover will happen
            mid_seg_parent1 = parent1[index_parent1:index_parent2+1]
            # last segment
            last_seg_parent1 = parent1[index_parent2+1:]
            ## child from all segments
            child1 = np.concatenate((first_seg_parent1,mid_seg_parent1,last_seg_parent1))                

            ### Parent 2
            # first segment
            first_seg_parent2 = parent2[:index_parent2]
            # middle segment; where the crossover will happen
            mid_seg_parent2 = parent2[index_parent2:index_parent2+1]
            # last segment
            last_seg_parent2 = parent2[index_parent2+1:]
            ## child from all segments
            child2 = np.concatenate((first_seg_parent2,mid_seg_parent2,last_seg_parent2))        
            return child1,child2
        #if probability logic is bypassed
        else:
            return parent1,parent2

    def _mutation(self,child,prob_mutation):

        # mutated child 1 placeholder
        mutated_child = np.empty((0,len(child)))

        ## get random probability at each index of chromosome and start with 0    
        t = 0
        for cld1 in child:
            rand_prob_mutation = np.random.rand() # do we mutate or no???
            # if random decimal generated is less than random probability, then swap value at index position
            if rand_prob_mutation < prob_mutation:
                # swap value
                if child[t] == 0:
                    child[t] = 1            
                else:
                    child[t] = 0
                # assign temporary child chromosome
                mutated_child = child

            #if random prob is >= mutation probability, assign as it is
            else:
                mutated_child = child

            # increase counter
            t = t+1
        return mutated_child
    
    def _getPopulationAndMatrix(self,doc_list,label_list,analyzer,min_df,max_df,stop_words,tokenizer,token_pattern,lowercase):
        #get null free df
        temp_df=pd.DataFrame({'doc_list':doc_list,'label_list':label_list})
        temp_df=temp_df[(~temp_df['doc_list'].isna()) & (~temp_df['label_list'].isna())]
        temp_df.reset_index(inplace=True,drop=True)
        label_list=temp_df['label_list'].tolist()
        doc_list=temp_df['doc_list'].tolist()
        del temp_df
        gc.collect()

        #get unique tokens
        tfidfvec = TfidfVectorizer(analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
        tfidfvec_vectorizer = tfidfvec.fit(doc_list)
        unique_words=list(tfidfvec_vectorizer.vocabulary_.keys())

        #count of tokens to consider based on percentage
        chromosome_to_feature = int(round((len(unique_words)/100)*self.percentage_of_token))

        #generate chromosome with number of 1 equal to percentage from total features specified by user
        population_array=np.concatenate([np.zeros(len(unique_words)-chromosome_to_feature),np.ones(chromosome_to_feature)])
        #shuffle after concatenating 0 and 1
        rd.shuffle(population_array)

        #create blank population matrix to append all individual chromosomes. number of rows equal to population size
        population_matrix = np.empty((0,len(unique_words)))

        #get population matrix
        population_matrix=self._get_population(self.population,population_matrix,population_array)

        #best solution for each generation
        best_of_a_generation = np.empty((0,len(population_array)+1))
        
        return doc_list,label_list,unique_words,population_array,population_matrix,best_of_a_generation

    def getGeneticFeatures(self,doc_list,label_list,model=LogisticRegression(),model_metric='f1',avrg='binary',analyzer='word',min_df=2,max_df=1.0,stop_words=None,tokenizer=None,token_pattern='(?u)\\b\\w\\w+\\b',lowercase=True):
        '''
        Data Parameters
        ----------        
        doc_list : text documents in a python list. 
            Example: ['i had dinner','i am on vacation','I am happy','Wastage of time']
        
        label_list : labels in a python list.
            Example: ['Neutral','Neutral','Positive','Negative']
        
        
        Modelling Parameters
        ----------
        model : Set a model which has .fit function to train model and .predict function to predict for test data. 
            This model should also be able to train classifier using TfidfVectorizer feature.
            Default is set as Logistic regression in sklearn
        
        model_metric : Classifier cost function. Select one from: ['f1','precision','recall'].
            Default is F1
        
        avrg : Averaging used in model_metric. Select one from ['micro', 'macro', 'samples','weighted', 'binary'].
            For binary classification, default is 'binary' and for multi-class classification, default is 'micro'.
        
        
        TfidfVectorizer Parameters
        ----------
        analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
            Whether the feature should be made of word or character n-grams.
            Option 'char_wb' creates character n-grams only from text inside
            word boundaries; n-grams at the edges of words are padded with space.
            
        min_df : float or int, default=2
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float in range of [0.0, 1.0], the parameter represents a proportion
            of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        max_df : float or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float in range [0.0, 1.0], the parameter represents a proportion of
            documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        stop_words : {'english'}, list, default=None
            If a string, it is passed to _check_stop_list and the appropriate stop
            list is returned. 'english' is currently the only supported string
            value.
            There are several known issues with 'english' and you should
            consider an alternative (see :ref:`stop_words`).

            If a list, that list is assumed to contain stop words, all of which
            will be removed from the resulting tokens.
            Only applies if ``analyzer == 'word'``.

            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.

        tokenizer : callable, default=None
            Override the string tokenization step while preserving the
            preprocessing and n-grams generation steps.
            Only applies if ``analyzer == 'word'``

        token_pattern : str, default=r"(?u)\\b\\w\\w+\\b"
            Regular expression denoting what constitutes a "token", only used
            if ``analyzer == 'word'``. The default regexp selects tokens of 2
            or more alphanumeric characters (punctuation is completely ignored
            and always treated as a token separator).

            If there is a capturing group in token_pattern then the
            captured group content, not the entire match, becomes the token.
            At most one capturing group is permitted.

        lowercase : bool, default=True
            Convert all characters to lowercase before tokenizing.        
        '''
        
        start = time.time()
        
        #define cost function averaging
        if len(set(label_list))>2:
            avrg='micro'
        else:
            avrg='binary'
        
        #get all parameters needed for GA
        doc_list,label_list,unique_words,population_array,population_matrix,best_of_a_generation=self._getPopulationAndMatrix(doc_list,label_list,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
        
        #Execute GA
        for genrtn in range(self.generations):
            
            ##if time exceeds then break loop
            if (time.time()-start)//60 > self.runtime_minutes:
                print('Run time exceeded allocated time. Producing best features generated so far:')
                break
            
            # placeholder for saving the new generation
            new_population = np.empty((0,len(population_array)))

            # placeholder for saving the new generation and obj func val
            new_population_with_obj_val = np.empty((0,len(population_array)+1))

            # placeholder for saving the best solution for each generation
            sorted_best = np.empty((0,len(population_array)+1))

            ## generate new set of population in each generation
            # each iteration gives 2 chromosome.
            # Doing it half the population size will mean getting matrix of population size equal to original matrix
            for family in range(int(self.population/2)):
                #get parents
                parent1,parent2=self._get_parents(population_array=population_array,population_matrix=population_matrix,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)

                #crossover
                child1,child2=self._crossover(parent1=parent1,parent2=parent2,prob_crossover=self.prob_crossover)

                #mutation
                mutated_child1=self._mutation(child=child1,prob_mutation=self.prob_mutation)
                mutated_child2=self._mutation(child=child2,prob_mutation=self.prob_mutation)

                #get cost function for 2 mutated child and print for generation, family and child
                cost1=self._computeFitness(gene=mutated_child1,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
                cost2=self._computeFitness(gene=mutated_child2,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)

                #create population for next generaion
                new_population = np.vstack((new_population,mutated_child1,mutated_child2))

                #save cost and child
                mutant1_with_obj_val = np.hstack((cost1,mutated_child1))
                mutant2_with_obj_val = np.hstack((cost2,mutated_child2))
                #stack both chromosome of the family
                new_population_with_obj_val = np.vstack((new_population_with_obj_val,mutant1_with_obj_val,mutant2_with_obj_val))

            #at end of the generation, change population as the stacked chromosome set from previous generation
            population_matrix=new_population

            ### find best solution for generation based on objective function and stack
            sorted_best = np.array(sorted(new_population_with_obj_val,key=lambda x:x[0],reverse=True))

            # print and stack
            print('Generation:',genrtn,'best score',sorted_best[0][0])
            best_of_a_generation = np.vstack((best_of_a_generation,sorted_best[0]))

        #sort by metric
        best_metric_chromosome_pair = np.array(sorted(best_of_a_generation,key=lambda x:x[0],reverse=True))[0]

        #best chromosome, metric and vocabulary
        best_chromosome=best_metric_chromosome_pair[1:]

        best_metric=best_metric_chromosome_pair[0]
        print('Best metric:',best_metric)

        best_vocabulary=list(np.array(unique_words)[list(map(bool,best_chromosome))])
        return best_vocabulary    
            
if __name__=="__main__":
    #Multiclass classification problem
    input_doc_list=['i am very happy','i just had an awesome weekend','this is a very difficult terrain to trek. i wish i stayed back at home.','i just had lunch','Do you want chips?']
    target=['Positive','Positive','Negative','Neutral','Neutral']
    fsOBJ=TextFeatureSelection(target=target,input_doc_list=input_doc_list)
    result_df=fsOBJ.getScore()
    print(result_df)


    #Binary classification
    input_doc_list=['i am content with this location','i am having the time of my life','you cannot learn machine learning without linear algebra','i want to go to mars']
    target=[1,1,0,1]
    fsOBJ=TextFeatureSelection(target=target,input_doc_list=input_doc_list)
    result_df=fsOBJ.getScore()

    print(result_df)


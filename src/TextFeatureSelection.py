#!/usr/bin/env python
# coding: utf-8

# In[413]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from collections import Counter

class TextFeatureSelection():
    '''
    target: list object which has categories of labels. for more than one category, no need to dummy code and instead provide label encoded values as list object.
    input_doc_list: List object which has text. each element of list is text corpus. No need to tokenize, as text will be tokenized in the module while processing. target and input_doc_list should have same length. 
    stop_words: Words for which you will not want to have metric values calculated. Default is blank
    metric_list: List object which has the metric to be calculated. There are 4 metric which are being computed as 'MI','CHI','PD','IG'. you can specify one or more than one as a list object. Default is ['MI','CHI','PD','IG'].    
    
    Chi-square(CHI), Mutual information(MI), Proportional difference(PD) and Information gain(IG) are 4 metric which are calculated for each tokenized word from the corpus to aid the user for feature selection.

    Chi-square and Mutual information criteria are developed using the method suggested in paper //A Comparative Study on Feature Selection in Text Categorization// by Yiming Yang and Jan O. Pedersen
http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E5CC43FE63A1627AB4C0DBD2061FE4B9?doi=10.1.1.32.9956&rep=rep1&type=pdf
and //Entropy based feature selection for text categorization// https://hal.archives-ouvertes.fr/hal-00617969/document by Christine Largeron, Christophe Moulin, Mathias Géry

    Proportional difference menthod is developed using method suggested in paper
//Categorical Proportional Difference: A Feature Selection Method for Text Categorization// by Mondelle Simeon, Robert J. Hilderman
https://pdfs.semanticscholar.org/6569/9f0e1159a40042cc766139f3dfac2a3860bb.pdf
and //Feature Selection and Weighting Methods in Sentiment Analysis// by Tim O`Keefe and Irena Koprinska
https://www.researchgate.net/publication/242088860_Feature_Selection_and_Weighting_Methods_in_Sentiment_Analysis


Chi-square:
     - It measures the lack of independence between t and c.
     - It has a natural value of zero if t and c are independent. If it is higher, then term is dependent
     - It is not reliable for low-frequency terms
     - For multi-class categories, we will calculate X^2 value for all categories and will take the Max(X^2) value across all categories at the word level.
     - It is not to be confused with chi-square test and the values returned are not significance values

Mutual information:
     - Rare terms will have a higher score than common terms.
     - For multi-class categories, we will calculate MI value for all categories and will take the Max(MI) value across all categories at the word level.

Proportional difference:
     - How close two numbers are from becoming equal. 
     - Helps ﬁnd unigrams that occur mostly in one class of documents or the other
     - We use the positive document frequency and negative document frequency of a unigram as the two numbers.
     - If a unigram occurs predominantly in positive documents or predominantly in negative documents then the PD will be close to 1, however if distribution of unigram is almost similar, then PD is close to 0.
     - We can set a threshold to decide which words to be included
     - For multi-class categories, we will calculate PD value for all categories and will take the Max(PD) value across all categories at the word level.

Information gain:
     - It gives discriminatory power of the word
    '''
    
    def __init__(self,target,input_doc_list,stop_words=None,metric_list=['MI','CHI','PD','IG']):
        self.target=target
        self.input_doc_list=input_doc_list
        self.stop_words=stop_words
        self.metric_list=metric_list
 
    def _ChiSquare(self,A,B,C,D,N):
        return (N*((A*D)-(C*B))**2)/((A+B)*(A+C)*(B+D)*(C+D))

    def _MutualInformation(self,A,B,C,N):
        return np.log((A*N)/((A+C)*(A+B)))
    
    def _InformationGain(self,A,B,C,D,N):
        return (-((A+C)/N)*np.log((A+C)/N))+(A/N)*np.log(A/(A+B))+(C/N)*np.log(C/(C+D))

    def _ProportionalDifference(self,A,B):
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
            aa=self._ProportionalDifference(A,B)
            out_df['Proportional Difference']=self._ProportionalDifference(A,B)
        if 'MI' in self.metric_list:
            aa=self._MutualInformation(A,B,C,N)
            out_df['Mutual Information']=self._MutualInformation(A,B,C,N)
        if 'CHI' in self.metric_list:
            aa=self._ChiSquare(A,B,C,D,N)
            out_df['Chi Square']=self._ChiSquare(A,B,C,D,N)
        if 'IG' in self.metric_list:
            aa=self._InformationGain(A,B,C,D,N)
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
                label_df_pd=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'PD_'+calc_base_label:result_dict[calc_base_label]['Proportional Difference']})
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
                label_df_mi=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'MI_'+calc_base_label:result_dict[calc_base_label]['Mutual Information']})
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
                label_df_chi=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'CHI_'+calc_base_label:result_dict[calc_base_label]['Chi Square']})
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
                label_df_ig=pd.DataFrame({'word list':result_dict[calc_base_label]['word list'],'IG_'+calc_base_label:result_dict[calc_base_label]['Information Gain']})
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


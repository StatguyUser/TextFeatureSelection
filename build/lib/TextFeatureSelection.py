#!/usr/bin/env python
# coding: utf-8

# In[413]:


from nltk import word_tokenize,sent_tokenize
import pandas as pd
import numpy as np

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
        
    def ChiSquare(self,A,B,C,D,N):
        return (N*((A*D)-(C*B))**2)/((A+B)*(A+C)*(B+D)*(C+D))

    def MutualInformation(self,A,B,C,N):
        return np.log((A*N)/((A+C)*(A+B)))
    
    def InformationGain(self,A,B,C,D,N):
        return (-((A+C)/N)*np.log((A+C)/N))+(A/N)*np.log(A/(A+B))+(C/N)*np.log(C/(C+D))

    def ProportionalDifference(self,A,B):
        return ((A-B)*(-1))/(A+B)
    
    def get_uniquewords(self):
        ##get unique words across all documents
        if self.stop_words:
            unique_words=[word for doc in self.input_doc_list for sent in sent_tokenize(doc) for word in word_tokenize(sent) if word not in self.stop_words]
        else:
            unique_words=[word for doc in self.input_doc_list for sent in sent_tokenize(doc) for word in word_tokenize(sent)]
        unique_words=set(unique_words)
        return unique_words
    
    def getvalues_singleclass(self,unique_words,calc_df):
        pd_val=[]
        mi=[]
        chi=[]
        ig=[]
        word_list=[]

        ##get base label for calculating values
        calc_base_label=list(set(self.target))[0]
        
        ##get binary pandas series for label if it is present row-wise or not
        label=calc_df['target']==calc_base_label

        for word in unique_words:
            try:
                
                #get binary pandas series for word if it is present row-wise or not
                word_presence=calc_df['input_doc_list'].str.contains('\\b'+word+'\\b')
                ##check if word count is existing and labels have value, to be sure if any regex error for word.
                if sum(word_presence) and sum(label):
                    cross_tab=pd.crosstab(label,word_presence)
                    A=cross_tab[1][True]
                    B=cross_tab[1][False]
                    C=cross_tab[0][True]
                    D=cross_tab[0][False]
                    N=A+B+C+D

                    if 'PD' in self.metric_list:
                        pd_val.append(self.ProportionalDifference(A,B))
                    if 'MI' in self.metric_list:
                        mi.append(self.MutualInformation(A,B,C,N))
                    if 'CHI' in self.metric_list:
                        chi.append(self.ChiSquare(A,B,C,D,N))
                    if 'IG' in self.metric_list:
                        ig.append(self.InformationGain(A,B,C,D,N))

                    word_list.append(word)
            except Exception as e:
                pass

        values_df=pd.DataFrame({'word list':word_list,'word occurence count':sum(word_presence)})

        if 'PD' in self.metric_list:
            values_df['Proportional Difference']=pd_val
        if 'MI' in self.metric_list:
            values_df['Mutual Information']=mi
        if 'CHI' in self.metric_list:
            values_df['Chi Square']=chi
        if 'IG' in self.metric_list:
            values_df['Information Gain']=ig
            values_df['Information Gain'].replace(np.nan,0,inplace=True)            

        return values_df
    
    def getvalues_multiclass(self,unique_words,calc_df):
        pd_val=[]
        mi=[]
        chi=[]
        ig=[]
        word_list=[]
        word_count=[]

        #get labels flag once in anther function, result as dict

        for word in unique_words:
            try:
                #category level calculation
                pd_val_cat=[]
                mi_cat=[]
                chi_cat=[]
                ig_cat=[]
                label=[]
                word_presence=[]

                for calc_base_label in set(self.target):
                    ##get binary pandas series for label if it is present row-wise or not            
                    label=calc_df['target']==calc_base_label
                    #get binary pandas series for word if it is present row-wise or not
                    word_presence=calc_df['input_doc_list'].str.contains('\\b'+word+'\\b')
                    ##check if word count is existing and labels have value, to be sure if any regex error for word.
                    if sum(word_presence) and sum(label):
                        cross_tab=pd.crosstab(label,word_presence)
                        A=cross_tab[1][True]
                        B=cross_tab[1][False]
                        C=cross_tab[0][True]
                        D=cross_tab[0][False]
                        N=A+B+C+D

                        #calculate category specific values
                        if 'PD' in self.metric_list:
                            pd_val_cat.append(self.ProportionalDifference(A,B))
                        if 'MI' in self.metric_list:
                            mi_cat.append(self.MutualInformation(A,B,C,N))
                        if 'CHI' in self.metric_list:
                            chi_cat.append(self.ChiSquare(A,B,C,D,N))
                        if 'IG' in self.metric_list:
                            ig_cat.append(self.InformationGain(A,B,C,D,N))

                #max value across all category for metric, based on specified metric
                if 'PD' in self.metric_list and pd_val_cat:
                    pd_val.append(np.ma.masked_invalid(pd_val_cat).max())

                if 'MI' in self.metric_list and mi_cat:
                    mi.append(np.ma.masked_invalid(mi_cat).max())

                if 'CHI' in self.metric_list and chi_cat:
                    chi.append(np.ma.masked_invalid(chi_cat).max())
                        
                if 'IG' in self.metric_list and ig_cat:
                    ig.append(np.ma.masked_invalid(ig_cat).max())

                ##if atleast one calculation list has value
                if pd_val_cat or mi_cat or chi_cat or ig_cat:
                    word_list.append(word)
                    word_count.append(sum(word_presence))
            except Exception as ex:
                pass

        values_df=pd.DataFrame({'word list':word_list,'word occurence count':word_count})
        if 'PD' in self.metric_list:
            values_df['Proportional Difference']=np.float64(pd_val)
        if 'MI' in self.metric_list:
            values_df['Mutual Information']=np.float64(mi)
        if 'CHI' in self.metric_list:
            values_df['Chi Square']=np.float64(chi)
        if 'IG' in self.metric_list:
            values_df['Information Gain']=np.float64(ig)
            values_df['Information Gain'].replace(np.nan,0,inplace=True)

        return values_df
    
    def getScore(self):
        if type(self.target)==list and type(self.input_doc_list)==list:            
            if len(self.target)!=len(self.input_doc_list):
                print('Please provide target and input_doc_list of similar length.')
            else:
                calc_df=pd.DataFrame({'target':self.target,'input_doc_list':self.input_doc_list})
                unique_words=self.get_uniquewords()
                if len(set(self.target))==2:
                    values_df=self.getvalues_singleclass(unique_words,calc_df)
                    return values_df
                elif len(set(self.target))>2:
                    values_df=self.getvalues_multiclass(unique_words,calc_df)
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


#!/usr/bin/env python
# coding: utf-8

# In[413]:


from nltk import word_tokenize,sent_tokenize
import multiprocessing as mp
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
       
    def _custom_cross_tab(self,label,word_presence):
        A_word=0
        B_word=0
        C_word=0
        D_word=0
        for i,j in zip(list(label),list(word_presence)):
            if i==True and j==True:
                A_word+=1
            elif i==False and j==False:
                D_word+=1
            elif i==True and j==False:
                C_word+=1
            elif i==False and j==True:
                B_word+=1
        #N=A+B+C+D
        return A_word,B_word,C_word,D_word
 
    def _ChiSquare(self,A,B,C,D,N):
        return (N*((A*D)-(C*B))**2)/((A+B)*(A+C)*(B+D)*(C+D))

    def _MutualInformation(self,A,B,C,N):
        return np.log((A*N)/((A+C)*(A+B)))
    
    def _InformationGain(self,A,B,C,D,N):
        return (-((A+C)/N)*np.log((A+C)/N))+(A/N)*np.log(A/(A+B))+(C/N)*np.log(C/(C+D))

    def _ProportionalDifference(self,A,B):
        return ((A-B)*(-1))/(A+B)
    
    def _get_uniquewords(self):
        ##get unique words across all documents
        if self.stop_words:
            unique_words=[word for doc in self.input_doc_list for sent in sent_tokenize(doc) for word in word_tokenize(sent) if word not in self.stop_words]
        else:
            unique_words=[word for doc in self.input_doc_list for sent in sent_tokenize(doc) for word in word_tokenize(sent)]
        unique_words=set(unique_words)

        return unique_words

    def _singleclass_Parallel(self,token,dataframe,label_array,results,token_presence_sum):

        try:
            token_presence=dataframe['input_doc_list'].str.contains('\\b'+token+'\\b')
            token_presence_sum=sum(token_presence)
            if token_presence_sum:
                results=self._custom_cross_tab(label_array,token_presence)
        except Exception as e:
            pass

        return results,token_presence_sum,token
    
    
    def _getvalues_singleclass(self,unique_words,calc_df):
        
        ##get base label for calculating values
        calc_base_label=list(set(self.target))[0]
        ##get binary pandas series for label if it is present row-wise or not
        label_array=calc_df['target']==calc_base_label
        
        ##parallel computation of ABCD
        pool=mp.Pool()
        results=[]
        token_presence_sum=0
        result = pool.starmap(self._singleclass_Parallel, [(words, calc_df,label_array,results,token_presence_sum) for words in unique_words])
        pool.close()

        ##unpacking ABCD, count and word
        result_single = [(*x,y,z) for x,y,z  in result if x] 
        unzp_lst = list(zip(*result_single))
        temp_df=pd.DataFrame({'A':unzp_lst[0],'B':unzp_lst[1],'C':unzp_lst[2],'D':unzp_lst[3],'word occurence count':unzp_lst[4],'word list':unzp_lst[5]})
        #get N
        temp_df['N']=temp_df['A']+temp_df['B']+temp_df['C']+temp_df['D']

        if 'PD' in self.metric_list:
            temp_df['Proportional Difference']=self._ProportionalDifference(temp_df.A,temp_df.B)
        if 'MI' in self.metric_list:
            temp_df['Mutual Information']=self._MutualInformation(temp_df.A,temp_df.B,temp_df.C,temp_df.N)
        if 'CHI' in self.metric_list:
            temp_df['Chi Square']=self._ChiSquare(temp_df.A,temp_df.B,temp_df.C,temp_df.D,temp_df.N)
        if 'IG' in self.metric_list:
            temp_df['Information Gain']=self._InformationGain(temp_df.A,temp_df.B,temp_df.C,temp_df.D,temp_df.N)
            temp_df['Information Gain'].replace(np.nan,0,inplace=True)
        
        ##get only computed columns
        filtered_cols=['word list','word occurence count','Proportional Difference','Mutual Information','Chi Square','Information Gain']
        for cols in filtered_cols[2:]:
            if cols not in temp_df.columns:
                filtered_cols.remove(cols)
        temp_df=temp_df[filtered_cols]
        return temp_df
    
    def _getvalues_multiclass(self,unique_words,calc_df):
        
        result_dict={}

        #for each class
        for calc_base_label in list(set(self.target)):
            ##get binary pandas series for label if it is present row-wise or not
            label_array=calc_df['target']==calc_base_label
            
            ##parallel computation of ABCD
            pool=mp.Pool()
            results=[]
            token_presence_sum=0
            result = pool.starmap(self._singleclass_Parallel, [(words, calc_df,label_array,results,token_presence_sum) for words in unique_words])
            pool.close()
            
            ##unpacking ABCD, count and word
            result_single = [(*x,y,z) for x,y,z  in result if x] 
            unzp_lst = list(zip(*result_single))
            temp_df=pd.DataFrame({'A':unzp_lst[0],'B':unzp_lst[1],'C':unzp_lst[2],'D':unzp_lst[3],'word occurence count':unzp_lst[4],'word list':unzp_lst[5]})
            #get N
            temp_df['N']=temp_df['A']+temp_df['B']+temp_df['C']+temp_df['D']            
            
            
            
            if 'PD' in self.metric_list:
                temp_df['Proportional Difference']=self._ProportionalDifference(temp_df.A,temp_df.B)
            if 'MI' in self.metric_list:
                temp_df['Mutual Information']=self._MutualInformation(temp_df.A,temp_df.B,temp_df.C,temp_df.N)
            if 'CHI' in self.metric_list:
                temp_df['Chi Square']=self._ChiSquare(temp_df.A,temp_df.B,temp_df.C,temp_df.D,temp_df.N)
            if 'IG' in self.metric_list:
                temp_df['Information Gain']=self._InformationGain(temp_df.A,temp_df.B,temp_df.C,temp_df.D,temp_df.N)
                temp_df['Information Gain'].replace(np.nan,0,inplace=True)
        
            ##get only computed columns
            filtered_cols=['word list','word occurence count','Proportional Difference','Mutual Information','Chi Square','Information Gain']
            for cols in filtered_cols[2:]:
                if cols not in temp_df.columns:
                    filtered_cols.remove(cols)
            temp_df=temp_df[filtered_cols]

            ##assign to dict for master calculation
            result_dict[calc_base_label]=temp_df
        
        ####merge
        final_results_pd=pd.DataFrame()
        final_results_mi=pd.DataFrame()
        final_results_chi=pd.DataFrame()
        final_results_ig=pd.DataFrame()
        
        #final result
        final_results=pd.DataFrame({'word list':temp_df['word list'],'word occurence count':temp_df['word occurence count']})
        
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
                calc_df=pd.DataFrame({'target':self.target,'input_doc_list':self.input_doc_list})
                unique_words=self._get_uniquewords()
                if len(set(self.target))==2:
                    values_df=self._getvalues_singleclass(unique_words,calc_df)
                    return values_df
                elif len(set(self.target))>2:
                    values_df=self._getvalues_multiclass(unique_words,calc_df)
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


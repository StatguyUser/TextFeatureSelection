What is it?
===========

TextFeatureSelection is a Python library which helps improve text classification models through feature selection. It has 2 methods `TextFeatureSelection` and `TextFeatureSelectionGA` methods respectively.

First method: TextFeatureSelection
=================
It follows the `filter` method for feature selection. It provides a score for each word token. We can set a threshold for the score to decide which words to be included. There are 4 algorithms in this method, as follows.

  - **Chi-square** It measures the lack of independence between term(t) and class(c). It has a natural value of zero if t and c are independent. If it is higher, then term is dependent. It is not reliable for low-frequency terms 
  - **Mutual information** Rare terms will have a higher score than common terms. For multi-class categories, we will calculate MI value for all categories and will take the Max(MI) value across all categories at the word level.
  - **Proportional difference** How close two numbers are from becoming equal. It helps ﬁnd unigrams that occur mostly in one class of documents or the other.
  - **Information gain** It gives discriminatory power of the word.

It has below parameters

  - **target** list object which has categories of labels. for more than one category, no need to dummy code and instead provide label encoded values as list object.
  - **input_doc_list** List object which has text. each element of list is text corpus. No need to tokenize, as text will be tokenized in the module while processing. target and input_doc_list should have same length. 
  - **stop_words** Words for which you will not want to have metric values calculated. Default is blank
  - **metric_list** List object which has the metric to be calculated. There are 4 metric which are being computed as 'MI','CHI','PD','IG'. you can specify one or more than one as a list object. Default is ['MI','CHI','PD','IG']. Chi-square(CHI), Mutual information(MI), Proportional difference(PD) and Information gain(IG) are 4 metric which are calculated for each tokenized word from the corpus to aid the user for feature selection.

How to use is it?
=================

```python

from TextFeatureSelection import TextFeatureSelection

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
```

Second method: TextFeatureSelectionGA
=================
It follows the `genetic algorithm` method. This is a population based metaheuristics search algorithm. It returns the optimal set of word tokens which give the best possible model score.

Its parameters are divided into 2 groups.

a) Genetic algorithm parameters: These are provided during object initialization.

  - **generations** Number of generations to run genetic algorithm. 500 as deafult, as used in the original paper
  - **population** Number of individual chromosomes. 50 as default, as used in the original paper
  - **prob_crossover** Probability of crossover. 0.9 as default, as used in the original paper
  - **prob_mutation** Probability of mutation. 0.1 as default, as used in the original paper
  - **percentage_of_token** Percentage of word features to be included in a given chromosome.
        50 as default, as used in the original paper.
  - **runtime_minutes** Number of minutes to run the algorithm. This is checked in between generations.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from generations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.

b) Machine learning model and tfidf parameters: These are provided during function call.

  Data Parameters

  - **doc_list** text documents in a python list. 
            Example: ['i had dinner','i am on vacation','I am happy','Wastage of time']
        
  - **label_list** labels in a python list.
            Example: ['Neutral','Neutral','Positive','Negative']
        
        
  Modelling Parameters

  - **model** Set a model which has .fit function to train model and .predict function to predict for test data. 
            This model should also be able to train classifier using TfidfVectorizer feature.
            Default is set as Logistic regression in sklearn
        
  - **model_metric** Classifier cost function. Select one from: ['f1','precision','recall'].
            Default is F1
        
  - **avrg** Averaging used in model_metric. Select one from ['micro', 'macro', 'samples','weighted', 'binary'].
            For binary classification, default is 'binary' and for multi-class classification, default is 'micro'.
        
        
  TfidfVectorizer Parameters

  - **analyzer** {'word', 'char', 'char_wb'} or callable, default='word'
            Whether the feature should be made of word or character n-grams.
            Option 'char_wb' creates character n-grams only from text inside
            word boundaries; n-grams at the edges of words are padded with space.
            
  - **min_df** float or int, default=2
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float in range of [0.0, 1.0], the parameter represents a proportion
            of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

  - **max_df** float or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float in range [0.0, 1.0], the parameter represents a proportion of
            documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

  - **stop_words** {'english'}, list, default=None
            If a string, it is passed to _check_stop_list and the appropriate stop
            list is returned. 'english' is currently the only supported string
            value.
            There are several known issues with 'english' and you should
            consider an alternative (see :ref:`stop_words`).
            If a list, that list is assumed to contain stop words, all of which
            will be removed from the resulting tokens.
            Only applies if analyzer == 'word'.
            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.

  - **tokenizer** callable, default=None
            Override the string tokenization step while preserving the
            preprocessing and n-grams generation steps.
            Only applies if analyzer == 'word'

  - **token_pattern** str, default=r"(?u)\\b\\w\\w+\\b"
            Regular expression denoting what constitutes a "token", only used
            if analyzer == 'word'. The default regexp selects tokens of 2
            or more alphanumeric characters (punctuation is completely ignored
            and always treated as a token separator).
            If there is a capturing group in token_pattern then the
            captured group content, not the entire match, becomes the token.
            At most one capturing group is permitted.

  - **lowercase** bool, default=True
            Convert all characters to lowercase before tokenizing.        

How to use is it?
=================

```python

from TextFeatureSelection import TextFeatureSelectionGA

#Input documents: doc_list
#Input labels: label_list

getGAobj=TextFeatureSelectionGA(percentage_of_token=60)
best_vocabulary=getGAobj.getGeneticFeatures(doc_list=doc_list,label_list=label_list)

```


Where to get it?
================

`pip install TextFeatureSelection`

Dependencies
============

 - [numpy](https://www.numpy.org/)

 - [pandas](https://pandas.pydata.org/)

 - [scikit-learn](https://scikit-learn.org/stable/)

 - [collections](https://docs.python.org/2/library/collections.html)

References
============

 - [A Comparative Study on Feature Selection in Text Categorization](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E5CC43FE63A1627AB4C0DBD2061FE4B9?doi=10.1.1.32.9956&rep=rep1&type=pdf) by Yiming Yang and Jan O. Pedersen
 - [Entropy based feature selection for text categorization](https://hal.archives-ouvertes.fr/hal-00617969/document) by Christine Largeron, Christophe Moulin, Mathias Géry
 - [Categorical Proportional Difference: A Feature Selection Method for Text Categorization](https://pdfs.semanticscholar.org/6569/9f0e1159a40042cc766139f3dfac2a3860bb.pdf) by Mondelle Simeon, Robert J. Hilderman
 - [Feature Selection and Weighting Methods in Sentiment Analysis](https://www.researchgate.net/publication/242088860_Feature_Selection_and_Weighting_Methods_in_Sentiment_Analysis) by Tim O`Keefe and Irena Koprinska
 - [Feature Selection For Text Classification Using Genetic Algorithms](https://ieeexplore.ieee.org/document/7804223) by Noria Bidi and Zakaria Elberrichi


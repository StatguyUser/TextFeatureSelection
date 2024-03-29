# What is it?
Companion library of machine learning book [Feature Engineering & Selection for Explainable Models: A Second Course for Data Scientists](https://statguyuser.github.io/feature-engg-selection-for-explainable-models.github.io/index.html).

TextFeatureSelection is a Python library which helps improve text classification models through feature selection. It has 3 methods `TextFeatureSelection`, `TextFeatureSelectionGA` and `TextFeatureSelectionEnsemble` methods respectively.

# First method: TextFeatureSelection
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

# How to use is it?
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

# Second method: TextFeatureSelectionGA
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

# How to use is it?
```python
from TextFeatureSelection import TextFeatureSelectionGA
#Input documents: doc_list
#Input labels: label_list
getGAobj=TextFeatureSelectionGA(percentage_of_token=60)
best_vocabulary=getGAobj.getGeneticFeatures(doc_list=doc_list,label_list=label_list)
```

# Third method: TextFeatureSelectionEnsemble

TextFeatureSelectionEnsemble helps ensemble multiple models to find best model combination with highest performance.

It uses grid search and document frequency for reducing vector size for individual models. This makes individual models less complex and computationally faster. At the ensemble learning layer, metaheuristics algorithm is used for identifying the smallest possible combination of individual models which has the highest impact on ensemble model performance.

Base Model Parameters

    
  - **doc_list** Python list with text documents for training base models
    
  - **label_list** Python list with Y labels

  - **use_class_weight** Boolean value representing if you want to apply class weight before training classifiers. Default is False.
    
  - **pickle_path** Path where base model, text feature vectors and ensemble models will be saved in PC.
    
  - **save_data** Boolean True | False. Default is False. Whether datasets used for training base model, and ensemble models will be saved in PC.
    
  - **n_crossvalidation** How many cross validation samples to be created. Higher value will result more time for model training. Lower number will result in less reliable model. Default is 5.
    
  - **seed_num** Seed number for training base models as well as for creating cross validation data. Default is 1.
    
  - **stop_words** Stop words for count and tfidf vectors. Default is None.
    
  - **lowercase** Lowercasing for text in count and tfidf vector. Default is True
    
  - **n_jobs** How many jobs to be run in parallel for training sklearn and xgboost models. Default is -1
    
  - **cost_function** Cost function to optimize base models. During feature selection using grid search for base models, this cost function is used for identifying which words to be removed based on combination of lower and higer document frequency for words.
  Available options are 'f1', 'precision', 'recall'. Default is 'f1'
    
  - **average** What averaging to be used for cost_function. Useful for multi-class classifications.
  Available options are 'micro','macro','samples','weighted' and 'binary'
  Default is 'binary'.
    
  - **basemodel_nestimators** How many n_estimators. Used as a parameter for tree based models such as 'XGBClassifier','AdaBoostClassifier','RandomForestClassifier','ExtraTreesClassifier'.
  Default is 500.

    
  - **feature_list** Type of features to be used for ensembling. Available options are 'Unigram','Bigram','Trigram'.
  Default is ['Unigram','Bigram','Trigram']
    
    
  - **vector_list** Type of text vectors from sklearn to be used. Available options are 'CountVectorizer','TfidfVectorizer'.
  Default is ['CountVectorizer','TfidfVectorizer']
    
    
  - **base_model_list** List of machine learning algorithms to be trained as base models for ensemble layer training.
  Available options are 'LogisticRegression','XGBClassifier','AdaBoostClassifier','RandomForestClassifier','ExtraTreesClassifier','KNeighborsClassifier'
  Default is ['LogisticRegression','XGBClassifier','AdaBoostClassifier','RandomForestClassifier','ExtraTreesClassifier','KNeighborsClassifier']
    
  
Metaheuristic algorithm feature selection parameters for ensemble model

  - **method** Which method you want to specify for metaheuristics feature selection. The available methods are 'ga', 'sa', 'aco', and 'pso'. These stand for genetic algorithm, simulated annealing, ant colony optimization, and particle swarm optimization respectively. You can select one out of the 4. Default is 'ga'. 

  - **MetaHeuristicsParameters** Parameters for the metaheuristics feature selection method for ensemble learning. This is used for identifying best combination of base models for ensemble learning. It helps remove models which has no contribution for ensemble learning and keep only important models.

  `FeatureSelection` module is used from `MetaHeuristicsFS` python library.
  Refer documentation for `MetaHeuristicsFS` at: https://pypi.org/project/MetaHeuristicsFS/ and example usage of MetaHeuristicsFS for feature selection: https://github.com/StatguyUser/feature_engineering_and_selection_for_explanable_models/blob/37ba0d2921fbabbb83df44c6eb7a1242b19a637f/Chapter%208%20-%20Hotel%20Cancelation%20.ipynb
  
  Parameters used are Parameters used are
```python
  {"model_object": LogisticRegression(n_jobs=-1,random_state=1),
  "cost_function":f1_score,
  "average":'micro',
  "cost_function_improvement":'increase',
  "ga_parameters":{"generations":50,
                  "population":50,
                  "prob_crossover":0.9,
                  "prob_mutation":0.1,
                  "run_time":120},
  "sa_parameters":{"temperature":1500,
                  "iterations":50,
                  "n_perturb":1,
                  "n_features_percent_perturb":1,
                  "alpha":0.9,
                  "run_time":120},
  "aco_parameters":{"iterations":50,
                  "N_ants":50,
                  "evaporation_rate":0.9,
                  "Q":0.2,
                  "run_time":120},
  "pso_parameters":{"iterations":50,
                  "swarmSize":50,
                  "run_time":120}
  }
```    
    
  Output are saved in 4 folders

    
  - **model** It has base models
    
  - **vector** it has count and tfidf vectors for each model
    
  - **ensemble_model** It has ensemble model
    
  - **deleted** It has base model and vectors for models which were discarded by genetic algorithm.
    
  - **data_files** It has list of data files used for training base models, and ensemble model
    
    Apart from above 5, it also saves and return list of columns which are used in ensemble layer with name best_ensemble_columns
    These columns are used in the exact same order for feature matrix in ensemble layer.

# How to use is it?
```python

imdb_data=pd.read_csv('../input/IMDB Dataset.csv')
le = LabelEncoder()
imdb_data['labels'] = le.fit_transform(imdb_data['sentiment'].values)

# convert raw text and labels to python list
doc_list=imdb_data['review'].tolist()
label_list=imdb_data['labels'].tolist()

# Initialize parameter for TextFeatureSelectionEnsemble and start training
gaObj=TextFeatureSelectionEnsemble(doc_list,label_list,n_crossvalidation=2,pickle_path='/home/user/folder/',average='micro',base_model_list=['LogisticRegression','RandomForestClassifier','ExtraTreesClassifier','KNeighborsClassifier'])
best_columns=gaObj.doTFSE()
```



# Where to get it?
`pip install TextFeatureSelection`

# How to cite
Md Azimul Haque (2022). Feature Engineering & Selection for Explainable Models: A Second Course for Data Scientists. Lulu Press, Inc.

# Dependencies
 - [pandas](https://pandas.pydata.org/)
 - [scikit-learn](https://scikit-learn.org/stable/)
 - [xgboost](https://xgboost.readthedocs.io/en/latest/)
 - [nltk](https://www.nltk.org/)
 - [MetaHeuristicsFS](https://pypi.org/project/MetaHeuristicsFS/)
 - [collections](https://docs.python.org/2/library/collections.html)

# References
 - [A Comparative Study on Feature Selection in Text Categorization](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E5CC43FE63A1627AB4C0DBD2061FE4B9?doi=10.1.1.32.9956&rep=rep1&type=pdf) by Yiming Yang and Jan O. Pedersen

 - [Entropy based feature selection for text categorization](https://hal.archives-ouvertes.fr/hal-00617969/document) by Christine Largeron, Christophe Moulin, Mathias Géry

 - [Categorical Proportional Difference: A Feature Selection Method for Text Categorization](https://pdfs.semanticscholar.org/6569/9f0e1159a40042cc766139f3dfac2a3860bb.pdf) by Mondelle Simeon, Robert J. Hilderman

 - [Feature Selection and Weighting Methods in Sentiment Analysis](https://www.researchgate.net/publication/242088860_Feature_Selection_and_Weighting_Methods_in_Sentiment_Analysis) by Tim O`Keefe and Irena Koprinska
 
 - [Feature Selection For Text Classification Using Genetic Algorithms](https://ieeexplore.ieee.org/document/7804223) by Noria Bidi and Zakaria Elberrichi
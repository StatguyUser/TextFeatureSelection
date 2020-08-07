What is it?
===========

TextFeatureSelection is a Python package providing feature selection for text tokens through filter method of feature selection and we can set a threshold to decide which words to be included. There are 4 methods for helping feature selection.

  - **Chi-square** It measures the lack of independence between term(t) and class(c). It has a natural value of zero if t and c are independent. If it is higher, then term is dependent. It is not reliable for low-frequency terms 

  - **Mutual information** Rare terms will have a higher score than common terms. For multi-class categories, we will calculate MI value for all categories and will take the Max(MI) value across all categories at the word level.

  - **Proportional difference** How close two numbers are from becoming equal. It helps ﬁnd unigrams that occur mostly in one class of documents or the other.

  - **Information gain** It gives discriminatory power of the word.

Input parameters
================

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

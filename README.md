# meta-sa
MAML for Sentiment Analysis
Abstarct:
Low-resource languages present a strong challenge for text classification tasks due to the small amount of
available data. Some languages do not have built in support in many applications. Models which rely on large
corpuses are ineffective in performing these tasks. Twitter is widely used globally and provides a rich source of
data for many languages. Often tweets will contain multilingual text as well, providing an additional challenge.
We propose tackling the problem of sentiment analysis on a variety of low-resource languages using meta-learning.
Our overall architecture involves a pipeline of a multilingual embedding model (LASER), a data augmentation
model (SuperGen), and model agnostic machine learning (MAML). We aim to predict both the sentiment as well
as the language from a variety of tweet datasets with provided ground truth annotations. These languages span
high resource languages, dialects of high resource languages, and low resource languages. Our findings indicate
that MAML, a common meta-learning method, may not be effective for 1-shot sentiment analysis of low resource
languages.


# embedding-projection
WIP, name is temporary

A project developing a potential technique for extracting information from contextual sentence embeddings (sBERT) 


### The projection seems to work decently:
![Projection of Reviews onto Sentiment Vector](./img/sentiment_distributions.png)

Words corresponding to the 10 smallest embeddings:
['worse', 'terrible', 'sucked', 'horrible', 'worst', 'bad', 'rotten', 'unacceptable', 'stupidity', 'awful']
Words corresponding to the 10 largest embeddings:
['pleasure', 'anytime', 'admired', 'admire', 'fabulous', 'classical', 'beloved', 'romantic', 'anthologies', 'lovely']



### To check if the annotation correlates with human annotators:
![Human Annotator Correlation with Semantic Projection](./img/Scatterplot_w_Person.png)

It seems there is a rather strong correlation between average human anotator and the projection method!



### Lets see get an idea of what an correlation baseline would even look like.
Annotations of semantics vary when on a continous scale. Different computational methods produce different scores, but different humans also correlate differently with eachother.

![Annotator Correlation](./img/Annotator_Corr.png)


## We now try to the method on a more complex setting such as Linguistic Acceptability
This metric can be found in the glue/cola test-set, formally defined as:
"*The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.*" 


Linguistic Acceptability is meant to be a harder test from the glue benchmark. 
**Acceptable sentences** are normal correct sentences such as:
"They drank the pub dry." --> label == 1

**Unacceptable sentences** are syntaxtically correct but makes no semantic sense, i.e.:
"They drank the pub." --> label == 0

It seems, *through visual investigation*, that the method isn't able to define a direction that encodes for this rather complex linguistic feature.

![Acceptability Histogram](./img/LA_validation_histogram.png)








# embedding-projection
WIP, name is temporary

A project developing a potential technique for extracting information from contextual sentence embeddings (sBERT) 


### It seems that the projection might have a hard time seperating sentiment from ?(Reviewness)
![Projection of Reviews onto Sentiment Vector](./img/sentiment_distributions.png)



Word projections tell the same story:
Words corresponding to the 10 smallest embeddings:
['rocky', 'bop', 'movies', 'tango', '##dance', '##hips', 'musicals', 'dances', 'carmen', 'comedies']
Words corresponding to the 10 largest embeddings:
['nazi', 'forge', 'covert', 'obscured', 'civilian', 'cockpit', 'wartime', 'pilot', 'sniper', 'darkly']



**Current hypothesis:**

The seperation became worse as I went from a subset of 500 --> 2000 movies.
I just chose the first 2000 movies. It might be they are ordered? I will to use a sample package for subset next.


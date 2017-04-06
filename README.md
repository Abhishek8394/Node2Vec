# Node2Vec
Node2Vec implementation based on paper by A. Grover and J. Leskovec. [Paper link](http://snap.stanford.edu/node2vec/).

This project generates walks based on the algorithm mentioned in the paper. Learning vectors from these walks will be implemented in next update.

Can first calculate all probabilities for AliasSampling in a file, but not used since it might take days to generate them all.

Current implementation generates the tables on a as needed basis and routinely cleans them up to avoid slowdown due to a large hashmap and also possibly running out of memory.
Can resume previous runs if program has to quit in the middle. 


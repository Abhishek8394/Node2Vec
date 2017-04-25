# Node2Vec
Node2Vec implementation based on paper by A. Grover and J. Leskovec. [Paper link](http://snap.stanford.edu/node2vec/).
The dataset used is BlogCatalog and can be found [here](http://socialcomputing.asu.edu/datasets/BlogCatalog3)

This project generates walks based on the algorithm mentioned in the paper. Learning vectors from these walks will be implemented in next update.

Can first calculate all probabilities for AliasSampling in a file, but not used since it might take days to generate them all.

Current implementation generates the tables on a as needed basis and routinely cleans them up to avoid slowdown due to a large hashmap and also possibly running out of memory.
Can resume previous runs if program has to quit in the middle. 

To run a complete experiment run following commands:
1. Compile `Main.Java`. Also make sure the dataset is stored in a directory and has appropriate paths set in `config.txt`. For using default configuration create a directory called `data` in same directory as the project and add `nodes.csv`, `groups.csv` and `groups-edges.csv` files obtained from BlogCatalog.

2. Run `java Main.Class <config_file_location>`. The main class takes just one command line parameter, the location of the configuration file. Look at [`config.txt`](https://github.com/Abhishek8394/Node2Vec/blob/master/config.txt) in main repository for more info. Make sure correct output directories for writing the walks exist along with proper permissions.

3. Go to the directory containing the [`embeddingLearner.py`](https://github.com/Abhishek8394/Node2Vec/blob/master/embedding/embeddingLearner.py). Create a directory called `runs` or with any name but it should match with `log_dir` in [`train_config.txt`](https://github.com/Abhishek8394/Node2Vec/blob/master/embedding/train_config.txt)

4. Run `python embeddingLearner.py --config-file=<config_file> --input-file=<walks_file>`. `config_file` is the location of configuration file for training the embedder. Look at [train_config.txt](https://github.com/Abhishek8394/Node2Vec/blob/master/embedding/train_config.txt) in the main repository for more info. The `walks_file` is the file to which the Java program wrote the output of its walks. The output of the runs will be stored in `log_dir\{timestamp}`. Use this location in next step.

5. Run `python embeddingLearner.py --load-embed-from=<directory of previous run> --vocabulary-size=<num_of_nodes_in_graph>`. The directory of run should be as mentioned in previous step. Vocabulary size is the number of nodes in the entire graph. This will create a file 'embeddings.txt' in the same folder provided in `load-embed-from` option.

6. Run `python onevAll.py --embedding-file=<embeddings.txt location> --meta-file=<embedding_config> --config-file=<classifier_config>`. `embedding-file` takes the location of "embeddings.txt" created in previous step. `meta-file` takes the value of config file to train the embeddings which will be used, set it to `train_config.txt` found in the same directory as the run for that embedding. `config-file` is configuration for training the classifier. It is in same file as the one used for embedding.

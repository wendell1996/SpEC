# SpEC
DASFAA 2020 "SpEC: Sparse Embedding-based Community Detection in Attributed Graphs"

# Dataset source
+ [LINQS](https://linqs.soe.ucsc.edu/data)
+ [SNAP](http://snap.stanford.edu/data/index.html)

# Environment
+ Python 3.6
+ `pip install -r requirements.txt`

# Example
+ Overlapping community detection: `/bin/bash example_overlapping.sh`
+ Non-overlapping community detection: `/bin/bash example_nonoverlapping.sh`

# Data format
+ `'adjacency_matrix'`: dense matrix or sparse matrix
+ `'labels_all'`: 0/1-dense matrix
+ `'features_all'`: dense matrix or sparse matrix
+ `'feat_dim'`: dimension of features
+ `'num_cate'`: number of categories
+ `'num_node'`: number of nodes

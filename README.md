# SpEC
DASFAA 2020 "SpEC: Sparse Embedding-based Community Detection in Attributed Graphs"

# Dataset source
+ [LINQS](https://linqs.soe.ucsc.edu/data) [1,2]
+ [SNAP](http://snap.stanford.edu/data/index.html) [3]

[1] Lu, Q., Getoor, L.: Link-based classiﬁcation. In: Proceedings of the 20th International Conference on Machine Learning (ICML-03). pp. 496–503 (2003)

[2] Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., Eliassi-Rad, T.: Collective classiﬁcation in network data. AI magazine 29(3), 93–93 (2008)

[3] Leskovec, J., Mcauley, J.J.: Learning to discover social circles in ego networks. In: Advances in neural information processing systems. pp. 539–547 (2012)

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

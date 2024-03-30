## Understanding Narrow Attention Mechanisms

### Projection of Inputs
Each input (or token) undergoes a projection process to create three sets of vectors: keys, values, and queries. These projections are crucial for computing attention scores and obtaining contextually relevant information.

### Chunking for Attention Heads
In models featuring multiple attention heads, the projections of keys, values, and queries are divided into smaller, disjoint subsets or chunks. Each attention head operates on its specific chunk, allowing the model to simultaneously focus on different aspects or representations of the input. This division aids in parallelizing computation and capturing diverse patterns in the input.

### Query Vector and Context Vector
The query vector, still a projection, serves as the reference for determining the attention scores across different chunks. It is compared with the keys to compute attention weights, which are then used to aggregate the corresponding values. The resulting context vector captures the weighted combination of information from different chunks, providing a contextually informed representation.

In essence, the chunking process in narrow attention mechanisms divides the input projections into smaller pieces to facilitate parallel processing and selective attention. The query vector guides this attention process, while the context vector consolidates information from different chunks to generate a comprehensive representation of the input.

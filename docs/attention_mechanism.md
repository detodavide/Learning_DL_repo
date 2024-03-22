## Attention Mechanism

- **Encoder Hidden States as Keys:** The encoder processes the input sequence and produces a set of hidden states, one for each time step. These hidden states represent the contextual information of the input sequence. In the context of attention mechanisms, these hidden states are treated as keys because they contain information that needs to be retrieved based on the current decoding step.

- **Decoder Hidden States as Queries:** The decoder generates the output sequence while attending to different parts of the input sequence. At each decoding step, the decoder's hidden state encapsulates the information generated so far in the decoding process. These hidden states are treated as queries in the attention mechanism because they represent the information the model is currently seeking or attending to.

- **Attention Scores:** The attention mechanism computes attention scores between the decoder's hidden state (query) and each of the encoder's hidden states (keys). These attention scores determine the importance or relevance of each encoder hidden state to the current decoding step.

- **Weighted Sum:** The attention scores are used to compute a weighted sum of the encoder's hidden states, where the weights are determined by the attention scores. This weighted sum represents the context vector, which is then combined with the decoder's hidden state to produce the final output at the current decoding step.

By treating the decoder's hidden states as queries and the encoder's hidden states as keys, the attention mechanism allows the model to focus on different parts of the input sequence dynamically as it generates the output sequence, effectively capturing the relevant information for each step of the decoding process.

After computing the attention scores, the next step in the attention mechanism is to multiply these scores by the values associated with the encoder's hidden states. This multiplication operation results in what is often referred to as the alignment vector or the context vector.

Here's a step-by-step explanation:

1. **Compute Attention Scores:** At each decoding step, the attention mechanism calculates attention scores between the decoder's hidden state (query) and each of the encoder's hidden states (keys). These attention scores represent how relevant each encoder hidden state is to the current decoding step.

2. **Softmax Normalization:** To ensure that the attention scores sum up to 1 and represent valid probabilities, a softmax function is applied to the attention scores. This step normalizes the scores across all encoder hidden states, turning them into attention weights.

3. **Compute Context Vector:** Once the attention scores are obtained and normalized, they are multiplied element-wise with the encoder's hidden states (values). This multiplication operation effectively weights each encoder hidden state based on its relevance to the current decoding step. The resulting vectors are then summed to produce the context vector or alignment vector.

Mathematically, this can be expressed as follows:

\[ \text{Context vector} = \sum_{i=1}^{N} \text{Attention score}_i \times \text{Encoder hidden state}_i \]

Where:
- \(N\) is the number of encoder hidden states.
- \(\text{Attention score}_i\) is the attention score associated with the \(i\)th encoder hidden state.
- \(\text{Encoder hidden state}_i\) is the \(i\)th encoder hidden state.

The context vector captures the most relevant information from the input sequence based on the attention mechanism's calculations. It provides the decoder with additional context for generating the output at the current decoding step.

Given a query vector \( Q \) and a set of key vectors \( K \), the dot product between \( Q \) and each key vector in \( K \) is computed. The dot product measures the similarity or relevance between the query and each key vector.

\[ \text{Attention score}_i = \text{softmax} \left( \frac{{Q \cdot K_i}}{{\sqrt{d_k}}} \right) \]

Where:
- \( Q \cdot K_i \) is the dot product between the query vector \( Q \) and the \(i\)th key vector \( K_i \).
- \( d_k \) is the dimensionality of the key vectors.
- The softmax function is applied to normalize the dot products, ensuring that the attention scores sum up to 1 and represent valid probabilities.

The softmax function transforms the dot products into a probability distribution, indicating the relative importance or attention assigned to each key vector. By dividing the dot products by \(\sqrt{d_k}\), where \(d_k\) is the dimensionality of the key vectors, the softmax operation is scaled to prevent the dot products from growing too large, which could result in gradients becoming too small during training (a problem known as vanishing gradients).

In summary, the softmax function applied to the dot products of the query and key vectors produces normalized attention scores, indicating the importance of each key vector relative to the query vector.

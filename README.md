# DRAGON
DRAGON is a device-cloud distributed RAG framework, which enables a simultaneous integration of personalized information and generic knowledge

## Terminology
|Term        |Description                                                         |
|------------|--------------------------------------------------------------------|
|passage     |the paragraph/chunk of text to be retrieved from assistant documents|
|context     |the passages inserted before the query                              |
|aggregate   |fusion/ensemble of multiple output distributions                    |
|embed       |transform the query/passages into vectors using the retriever model |
|encode      |transform raw text into tokens                                      |
|decode      |transform tokens into raw text                                      |
|generate    |predict the output sequence given the query                         |
|rag         |predict the output sequence using generator, given the query concatenated with context retrieved by the retriever|
|s_*(single) |size/length of *                                                    |
|n_*(plural) |number of *                                                         |
|bs_*(single)|batch size of *                                                     |
## Aggregation
$$\begin{aligned}
log(y)&=\log \sum_{k} \frac{\exp(w^k)}{\sum_i \exp(w^i)}\cdot \frac{\exp(z^k)}{\sum_j \exp(z^k_j)}\\
&=\log \sum_k \exp(\log \frac{\exp(w^k)}{\sum_i \exp(w^i)}+\log \frac{\exp(z^k)}{\sum_j \exp(z^k_j)})\\
&=log\\_sum\\_exp(log\\_softmax(w)+log\\_softmax(z))
\end{aligned}$$
## Language Modeling
|Generator  |Retriever  |Dataset   |Ensemble|BPB    |
|-----------|-----------|----------|--------|-------|
|opt-1.3b   |Contriever |wikitext-2|0       |2.8867 |
|           |           |          |2       |2.7041 |
|           |           |          |4       |2.6968 |
|           |           |          |4_concat|2.7070 |
|           |           |          |10      |2.6899 |
|           |           |          |16      |       |
## Sequence-wise vs. Token-wise
| query | output_seq | output_tok |
| --- | --- | --- |
| I love China, because | it is the people . I love of the world . love the food . | it's so different from the rest of the world. I love the way |
| when did us get involved in vietnam war | ?\n\n been fighting in vietnam in Vietnam than the yearss | ?\n\nThe United States entered the Vietnam War in 1965, when the |

## TODO List
### Logic
- [ ] Revise Replug to support token-wise aggregation
- [ ] Dynamic document weight
- [ ] Distributed RAG
- [ ] Speculative DRAG
- [ ] Scheduling algorithm

### Optimization
- [ ] Batch the output ensemble
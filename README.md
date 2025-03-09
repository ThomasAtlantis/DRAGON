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
## Aggregation Algorithm
$$\begin{aligned}
log(y)&=\log \sum_{k} \frac{\exp(w^k)}{\sum_i \exp(w^i)}\cdot \frac{\exp(z^k)}{\sum_j \exp(z^k_j)}\\
&=\log \sum_k \exp(\log \frac{\exp(w^k)}{\sum_i \exp(w^i)}+\log \frac{\exp(z^k)}{\sum_j \exp(z^k_j)})\\
&=log\\_sum\\_exp(log\\_softmax(w)+log\\_softmax(z))
\end{aligned}$$

## Experiment: Language Modeling
### Hyperparameters
|Hyperparameter         |Value   |Description                           |
|-----------------------|--------|--------------------------------------|
|retriever.s_passage    |64      |Number of words in a retrieved passage|
|text.with_title        |true    |Insert title before each passage      |
|generator.s_sequence   |896     |Total tokens in a test rolling window |
|evaluator.s_prefix     |128     |Total tokens as query for retrieval   |

### Results
|Generator  |Retriever  |Dataset     |s_context |n_docs    |s_aggregate|Bits-per-Byte|
|-----------|-----------|------------|----------|----------|-----------|-------------|
|opt-1.3b   |Contriever |wikitext-2  |128       |0         |0          |2.7324       |
|           |           |            |128       |1         |1          |2.7208       |
|           |           |            |128       |2         |2          |2.7100       |
|           |           |            |128       |4         |4          |2.7030       |
|           |           |            |128       |8         |8          |2.6986       |
|           |           |            |128       |16        |16         |2.6953       |
|           |           |            |128       |4         |1          |2.7148       |
|           |           |            |512       |4         |1          |2.7049       |
|           |           |            |1024      |4         |1          |2.7049       |
|           |           |            |128       |4         |2          |2.7032       |
|           |           |            |512       |4         |2          |2.6999       |
|           |           |            |512       |8         |2          |2.6870       |
|           |           |wikitext-103|128       |0         |0          |2.7324       |
|           |           |            |128       |1         |1          |2.7178       |
|           |           |            |128       |2         |2          |2.7040       |
|           |           |            |128       |4         |4          |2.6932       |
|           |           |            |128       |8         |8          |2.6829       |
|           |           |            |128       |16        |16         |2.6738       |
|           |           |            |128       |4         |1          |2.7079       |
|           |           |            |512       |4         |1          |2.6881       |
|           |           |            |1024      |4         |1          |2.6881       |
|           |           |            |128       |4         |2          |2.6913       |
|           |           |            |512       |4         |2          |2.6866       |
|           |           |            |512       |8         |2          |2.6612       |

**Average Number of Tokens in Document Context before Clipping**
|n_docs    |s_aggregate|avg_doc_len|
|----------|-----------|-----------|
|0         |0          |0          |
|1         |1          |66.65      |
|2         |2          |72.70      |
|4         |4          |77.11      |
|4         |1          |264.91     |
|4         |2          |142.86     |
|8         |8          |80.38      |
|8         |2          |280.12     |
|16        |16         |83.55      |

## Experiment: Sequence-wise vs. Token-wise
### Hyperparameters
|Hyperparameter          |Value   |Description                           |
|------------------------|--------|--------------------------------------|
|retriever.s_passage     |64      |Number of words in a retrieved passage|
|text.with_title         |true    |Insert title before each passage      |
|retriever.passages      |Salesforce/wikitext,wikitext-2-raw-v1|Dataset as passage corpus|
|retriever.s_context     |128|Maximum number of tokens allowed for passage contexts|
|retriever.n_docs        |4|Number of retrieved passages|
|retriever.s_aggregate   |4|Number of parallel decoding processes|
|generator.model         |facebook/opt-1.3b|LLM|
|evaluator.max_new_tokens|100|Number of tokens in the output sequence|

### Results
|query |output_seq |output_tok|
|------|-----------|----------|
| Carbon dioxide is | a greenhouse gas is gas the Earth, viewed as a thatant global gas. is is viewed as a pollutant of to it is also viewed as a pollutant . Thus it is also viewed as a pollutant is Thus it is also . as a pollutant . is it is also viewed as a pollutant . that it is also viewed as a pollutant. gas it is also viewed as a pollutant . a it is also viewed a pollutant . is it is also viewed as | a greenhouse gas that is produced by the burning of fossil fuels. It is a major component of the atmosphere and is responsible for the greenhouse effect. It is also a greenhouse gas that is produced by the burning of fossil fuels. It is a major component of the atmosphere and is responsible for the greenhouse effect. It is also a greenhouse gas that is produced by the burning of fossil fuels. It is a major component of the atmosphere and is responsible for the greenhouse effect. It is also a greenhouse gas that|
| Ernest Miller Hemingway was | a in writerveston's 18 also a cited as. to of the investon acht Club, of he of also a themervestoningway Club who was a great writer theing he wason aacht man. He was a man who the a greaton, the he was also a great man of the was aon who was a great line of the novel was the a greaton Y line was a novel is was a first line of the novel is also a | a writer and poet who was born in Galveston, Texas in 1882. He was the son of a Methodist minister and a schoolteacher. He was educated at the University of Texas and the University of Texas at Austin. He was a member of the Sigma Phi Epsilon fraternity. He was a member of the Texas Writers Club and the Texas Writers Club. He was a member of the Texas Writers Club. He was a member of the Texas Writers Club. He was a member of the |

**Augmented with Token-wise Reranking**
|query |output_seq |output_tok|
|------|-----------|----------|
|Carbon dioxide is| a greenhouse gas is gas the Earth, viewed as a thatant global gas. is is viewed as a pollutant of to it is also viewed as a pollutant . Thus it is also viewed as a pollutant is Thus it is also . as a pollutant . is it is also viewed as a pollutant . that it is also viewed as a pollutant . gas it is also viewed as a pollutant . a it is also viewed\n a pollutant . is it is also viewed as| a greenhouse gas that is produced by the burning of fossil fuels. It is a major component of the atmosphere and is responsible for the greenhouse effect. It is also a greenhouse gas that is produced by the burning of fossil fuels. It is a major component of the atmosphere and is responsible for the greenhouse effect. It is also a greenhouse gas that is produced by the burning of fossil fuels. It is a major component of the atmosphere and is responsible for the greenhouse effect. It is also a greenhouse gas that|
|Ernest Miller Hemingway was| a in writerveston's 18 also a cited as. to of the investon\nacht Club, of he of also a themervestoningway Club who was a great writer theing he wason aacht man. He was a man who the a greaton, the he was also a great man of the was aon who was a great line of the novel was the a greaton Y line was a novel is was a first line of the novel is also a\n| a writer and poet who was born in Galveston, Texas in 1882. He was the son of a Methodist minister and a schoolteacher. He was educated at the University of Texas and the University of Texas at Austin. He was a member of the Sigma Phi Epsilon fraternity. He was a member of the Texas Writers Club and the Texas Writers Club. He was a member of the Texas Writers Club. He was a member of the Texas Writers Club. He was a member of the|

## TODO List
- [ ] Set top-k sampling to reduce the size of transmission data
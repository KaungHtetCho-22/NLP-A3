# NLP-A3
NLP assignment from AIT

**Kaung Htet Cho (st124092)**

## Table of Contents
1. [Task1](#Task1)
2. [Task2](#Task2)
3. [Task3](#Task3)

## Task1
#### Dataset Acquisition

Collect Eng / Myanmar parallel copurs from **TUFS Asian Language Parallel Corpus (TALPCo)** (https://github.com/matbahasa/TALPCo). Combine these two .txt files and make english-mynamar pairs. And then uploaded to hugging face repository in order to make load_datasets easier. (https://huggingface.co/datasets/KaungHtetCho/MT-myanmar-english)

*Parameters*  ==>  (train = 1093 rows | test  = 137 rows | valid = 137 rows)

#### Preprocessing

Referenced by **Dr.Ye Kyaw Thu** myWord repo (https://github.com/ye-kyaw-thu/myWord) for Burmese language tokenization. 

- *Input* | Start with Burmese text and then it is preprocessed to remove spaces because spaces are not consistently used as word delimiters in Burmese language

- *Viterbi Algorithm* | Set the recursion limit to handle long sequences of text. Load unigram and bigram dictionaries from binary files.

- *Unigram and Bigram* | 'Class ProbDist' reads a dictionary file (either unigram or bigram) and calculates the probability of each word or word pair because it is crucial for Viterbi in deciding where to segment the text

- *Segmentation with Viterbi Algorithm* | Split the text into all possible first words and remaining text pairs.

- *Calculate probabilities for each segment* | Calculate the probability of the first word given its previous word (initially  for the start of the sentence) using the bigram model. Recursively call the Viterbi function on the remaining text to segment it further.

- *Output* | Return the list of tokenized words as the output of the my_tokenizer function.

## Task2
#### Experiment with attention mechanisms

| Attentions          | Training Loss | Training PPL | Validation Loss | Validation PPL | 
|----------------|-------------|---------------|---------------|--------------------|
| General Attention       |    2.077      |      7.980  |       2.971        |            19.519        |   
| Multiplicative Attention |          2.081   |     8.013      |         2.954   |          19.187          |    
| Additive Attention          |    2.049        |  7.761   |        2.949 |         19.087        |    

## Task3
#### 1. Performance comparisons

- epochs = 30
- device = Nvidia GeForce RTX 3060
- batch_size = 64
- num_heads = 8
- num_layers = 3
- optimizer = Adam
- learning_rate = 0.0001

| Attentions          | Training time | Test PPL |
|----------------|-------------|---------------|
| General Attention       |    7m 10s        |   18.541     |    
| Multiplicative Attention |          5m 50s   |     19.520       |  
| Additive Attention          |       7m 23s    |    17.700      |  

#### 2. Performance plots

<table>
  <tr>
    <td align="center">
      <img src="./app/images/general_attention_transformer_loss_plot.png" alt="Seq2seq transformer with general attention" style="width: 100%;" />
      <br />
      <em>Fig. 1: Seq2seq transformer with general attention</em>
    </td>
    <td align="center">
      <img src="./app/images/multiplicative_attention_transformer_loss_plot.png" alt="Seq2seq transformer with multiplicative attention" style="width: 100%;" />
      <br />
      <em>Fig. 2: Seq2seq transformer with multiplicative attention</em>
    </td>
    <td align="center">
      <img src="./app/images/additive_attention_transformer_loss_plot.png" alt="Seq2seq transformer with additive attention" style="width: 100%;" />
      <br />
      <em>Fig. 3: Seq2seq transformer with additive attention</em>
    </td>
  </tr>
</table>

#### 3. Attention maps

<table>
  <tr>
    <td align="center">
      <img src="app/images/general_attention_map.png" alt="general attention map" style="width: 100%;" />
      <br />
      <em>Fig. 1: General attention map</em>
    </td>
    <td align="center">
      <img src="app/images/multiplicative_attention_map.png" alt="multiplicative attention map" style="width: 100%;" />
      <br />
      <em>Fig. 2: Multiplicative attention map</em>
    </td>
    <td align="center">
      <img src="app/images/additive_attention_map.png" alt="additive attention map" style="width: 100%;" />
      <br />
      <em>Fig. 3: Additive attention map</em>
    </td>
  </tr>
</table>


#### 4. Analysis report

Implemented Human Evaluation


## Task3
### Web app documentation

The Website can be accessed on http://localhost:8000. User can type (English sentence) to input box and then the model translate into (Burmese sentence). My model is trained on transformer using general attention, multiplicative attention, additive attention mechanism. The main purpose is to see the differences between those mechanisms (The translational accuracy). According to the loss and perplexity scores, you can see the addictive model get the most similar accuracy with the true burmese meaning
# SMM

Semantic Matching Model with the word-to-sentence interaction mechanism and multi-task learning for Duplicate Question Detection.

## Data and code for:

Xu Z, Yuan H. Forum Duplicate Question Detection by Domain Adaptive Semantic Matching[J]. IEEE Access, 2020, 8: 56029-56038.

### Requirements

- Python 2.7
- [TensorFlow](https://www.tensorflow.org)
- [Scikit-Learn](http://scikit-learn.org/stable/index.html)
- [Numpy](http://www.numpy.org/)
- [jieba](https://pypi.org/project/jieba/)(For Chinese data set)

### Files

[preprocess.py]() for data preprocess

[Semantic_Matching.py]() is the Semantic Matching Model

 [train_smm.py]() is the code for training the Semantic Matching Model

### Prepare for run

#### Word embedding

The dimension of both Chinese and English word embedding are set to be 300. You need to download the pre-trained 300-D Chinese word vector (e.g [pre-trained w2v](<https://github.com/Embedding/Chinese-Word-Vectors>)) and 300-D English word vector (e.g [pre-trained glove](<https://nlp.stanford.edu/projects/glove/>)) , put them into your local path and revise the file name in line 111 and line 136 of  [train_smm.py]() 

#### The Format of the Data Set

Source and target data set are both csv file with the format showed in [ml.csv]()

### Run

#### For English Data Set

```python
python train_smm.py -s [the path of the source data set] -t [the path of the target data set]
```

e.g

```python
python train_smm.py -s ../data/stackexchange/stats.csv -t ../data/stackexchange/mathematica.csv
```

#### For Chinese Data Set

```python
python train_smm.py -s [the path of the source data set] -t [the path of the target data set] -c 1
```

e.g

```python
python train_smm.py -s ../data/stackexchange/stats.csv -t ../data/stackexchange/mathematica.csv -c 1
```

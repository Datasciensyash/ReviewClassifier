# ReviewClassifier

Model and api to classifiy film reviews on 10 point scale.

---

## Dataset

### Dataset source
For training and evaluating model was chosen [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), gathered at [IMDB](https://www.imdb.com/). Dataset contains 50,000 user film reviews with positive or negative label. 25,000 in the test and training part.

Here a small example of dataset samples:
[EX]

### Data preparation
Dataset was originally stored in 50,000 .txt files, but during the data preparation process was created pandas `DataFrame`'s with train and test samples. Also, i have performed the deletion of not-unique samples from dataset, which reduced size of train and test 24,904 and 24,801 respectively.

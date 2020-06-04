# ReviewClassifier

A model and api to classify movie reviews as negative or positive and then map classification results to 10-point scale.

---

## Dataset

#### Dataset source
Model training and evaluation were carried out on [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), gathered from [IMDB](https://www.imdb.com/).  
Dataset contains `50,000` movie reviews with labels (positive or negative).
Dataset is balanced: 50% of labels are positive and 50% of labels are negative.
Movie reviews were divided up into three subsets - train, val, test - with following ratios: 40% train, 10% val, 50% test stratified by label.

Here a small example of dataset samples:

|Target|Text                                                                                                                                                                                                       |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|0     |There was a time when Joel Schumacher was ranked quite high on my list of favorite directors. Back in the late 80's and early 90's, when his name was attached to several great films like "The Lost Boy...|
|1     |This little picture succeeds where many a big picture fails. Because it was a little picture, John Ford was not harassed by the studio big wigs. He was happier with this film than any other because he...|
|0     |Just watched on UbuWeb this early experimental short film directed by William Vance and Orson Welles. Yes, you read that right, Orson Welles! Years before he gained fame for radio's "The War of the Wo...|
|0     |What a waste of great acting talent. This is a shame because with Catherine Deneuve, Mathieu Amalric, Emmanuelle Devos, Chiara Mastroianni, and Melvil Poupaud (not to mention others less well known in...|
|0     |This movie plays out like an English version of an ABC after school special, with nudity. It makes you wonder who the target audience was supposed to be. It's not as though the writers were too preocc...|
|1     |I first saw this movie at least thirty years ago, and it remains one of my all-time faves! It's a classic - the intriguing plot, great characters, suspense and shocking twist ending (all set against t...|
|1     |I totally disagree with the comments of one of the critics before me who bashed the film. Having read the book, being impressed by it although this is a kind of literature that you cannot really LIKE ...|


#### Data preparation
Dataset was originally stored in `50,000` `.txt` files.
They were merged and converted to `pandas.DataFrame`s with train and test samples. Also, I've dropped non-unique samples from dataset, which reduced the size of train and test sets to `24,904` and `24,801` respectively. These files are stored in `./dataset/` directory (in `.pkl` format).

---

## Model and training

#### Model

Model consists of two components:
- `TF-IDF vectorizer` for text vectorization
- `Logistic Regression` for vector-based classification

#### Training Process

Baseline was to simply combine `sklearn`'s `TfidfVectorizer` and `sklearn`'s `LogisticRegression` with following hyperparameters:

```yaml
Vectorizer:
  - min_df: 1
  - max_df: 0.9
LogisticRegression:
  - solver: lbfgs
  - max_iter: 100
```
That gave me about `0.89` accuracy and `0.3` LogLoss on test data. 
I decided to try hyperparameter optimization with the following tricks:
- usage of unsupervised data (sometimes it gives better metrics values)
- usage of re-training on pseudo-labeled data.

For hyperparameter optimization I've chosen W&B Sweeps platform. All **results**, **logs** and **charts** are publicly available at [project page on W&B](https://app.wandb.ai/datasciensyash/review_classifier/sweeps/u3l9ojto/overview?workspace=user-datasciensyash). Also, this information is stored at `./csv/Hyperoptinfo.csv` file. Training with best hyperparameters has given me `0.89` accuracy and `0.28` LogLoss on test data.

Best hyperparameters have been:
```yaml
Vectorizer:
  - min_df: 1
  - max_df: 0.9518
LogisticRegression:
  - max_iter: 106
  - solver: sag
Strategy:
  - vectorizer_fit_unsup: False #Do we need to fit vectorizer with unsupervised data
  - pseudolabel_unsup: True #Do we need to perform pseudo-labeling.
```

![Hyperparameters optimization](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/wandb.png)

This big chart represents all used hyperparameter sets with their validation scores (on the left).

---

## Rating model

The main goal of the project wasn't `positive`/`negative` classification, I wanted to **map** classification **predictions to movie rating**. A naive way to do that is to multiply the probability of `positive` class by `9` and add `1`. But it doesn't work correctly because of the distribution of predictions (see below). 

Histogram below represents the distribution of film ratings at imdb (ratings are from [this dataset](https://raw.githubusercontent.com/miptgirl/kinopoisk_data/master/kp_all_movies.csv)). 
My dataset consists of `positive` and `negative` reviews, or only reviews with rating `> 7` and `< 4` accordingly. `Neutral` class is not present in my dataset.

![D-1](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/distribution-1.png)

When we map our `positive`-class probability to the rating we want to get similar distribution in production. But re-scaling probability to `1-10` range produces something undeniably different. As we can see at histogram below, our model assigns either probability values too close to `1` or too close to `0`. The colors here are for comparison only. For example, the share of the negative class in the dataset described above is approximately `10%`. Red color shows `10%` of the distribution of predictions. And so on.

Histogram of model predictions distribution:

![D-2](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/distribution-2.png)

Histogram of model predictions distribution after rescaling:

![D-3](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/distribution-3.png)

So we need to create a mapping from this distribution to initial movie rating distribution from imdb, e.g. by modified  `Inverse Transform Sampling` method (Implementation is in `./modules/dist_map.py`), using a small number of bins for a smoother plot. `x` here represents rescaled predictions axis, and `y` - rating axis. As we can see, that is exactly what we were looking for: this function reduces the likelihood of assigning an extremely low or extremely high rating to the film. 

![Map](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/map.png)

---

## Testing Rating model

To test the rating model I have scraped `700` reviews from imdb with their scores. This small dataset is located in `./csv/review_ratings.csv`. Plot below shows predictions two mapping types: simple `rescaling`(`model_prediction * 9 + 1`) and `rescaling`(`model_prediction * 9 + 1`) + `mapping`(`./modules/dist_map.py`) described above.  

Red line shows ground truth labels. 

As you can see, simple rescaling works worse.

You can see code for gathering dataset and metric evaluation process in `./Movie Rating Test.ipynb`.

|Method   |MSE_1 |MSE_2|MSE_3|MSE_4|MSE_5|MSE_6|MSE_7|MSE_8|MSE_9|MSE_10|Mean  |
|---------|------|-----|-----|-----|-----|-----|-----|-----|-----|------|------|
|Rescaling|4.771 |5.009|5.871|6.201|9.084|7.536|6.369|3.731|1.931|4.072 | 5.443|
|Mapping  |12.504|8.75 |5.421|2.786|2.565|1.603|1.328|1.263|2.697|6.537 | 4.423|


![Rating test](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/dist_compare.png)

As you can see, this **dataset** (which contains `700` reviews) is well balanced, which does **not correspond to real data**, therefore, `MSE` of the mapping model has been overestimated, as well as rescaling method has been underestimated.

---

## Deploy and API

Model has been [deployed](https://filmreviewclassifier.herokuapp.com/model_handler/?input=None) on Django at heroku. For details, see `./api/` folder. 

Usage example:
```python
import requests
url = 'https://filmreviewclassifier.herokuapp.com/model_handler/'
review = 'This film is so boring, i just fall asleep'
requests.get(url, {'input': review})

>> {"Predictions": [{"Class": -1, "Description": "Negative", "Rating": 1.3, "Rating_rounded": 1}]}
```
Response fields:
- `Class`: Review class (откуда - из модели или с imdb?)
  - `-1` stands for negative
  - `0`  stands for neutral
  - `1`  stands for positive
- `Description`: class description
- `Rating`: predicted rating.
- `Rating_rounded`: rounded predicted rating

---


## Pretrained models

Pretrained models are stored in `./models/` directory.
- `model.pkl`: `LogisticRegression` model
- `vectorizer.pkl`: `TfidfVectorizer` model
- `rating.pkl`: `DistributionMap` (`./modules/dist_map.py`) model


---

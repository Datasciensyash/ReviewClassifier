# ReviewClassifier

Model and api to classifiy film reviews on 10 point scale.

---

## Dataset

#### Dataset source
For training and evaluating model was chosen [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), gathered at [IMDB](https://www.imdb.com/). Dataset contains `50,000` user film reviews with positive or negative label. `25,000` in the test and training part. All two classes are completely balanced, `12,500` per positive and negative class.

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
Dataset was originally stored in `50,000` `.txt` files, but during the data preparation process was created pandas `DataFrame`'s with train and test samples. Also, i have performed the deletion of not-unique samples from dataset, which reduced size of train and test sets to `24,904` and `24,801` respectively.

---

## Model and training

#### Model

Model was composed of two components:
- `TF-IDF vectorizer` for text vectorization
- `Logistic Regression` for vector classification

#### Training Process

Baseline was to simple combine `sklearn`'s `TfidfVectorizer` and `sklearn`'s `LogisticRegression` with following hyperparameters:

```yaml
Vectorizer:
  - min_df: 1
  - max_df: 0.9
LogisticRegression:
  - solver: lbfgs
  - max_iter: 100
```
That achieved about `0.89` accuracy and `0.3` LogLoss on Test data. But Baseline hasn't used unsupervised data to train, which in theory can get better metrics value. Also, baseline hasn't used strategy of re-training on pseudo-labeled data.

For hyperparameter optimization was used W&B Sweeps platform. All **results**, **logs** and **charts** public avaliable at [project page on W&B](https://app.wandb.ai/datasciensyash/review_classifier/sweeps/u3l9ojto/overview?workspace=user-datasciensyash). Also, this information is stored at `./csv/Hyperoptinfo.csv` file. Training with best hyperparameters gain `0.89` accuracy and `0.28` LogLoss on Test data.

![Hyperparameters optimization](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/wandb.png)

---

## Rating model

In this project the main goal was not to classify `positive` and `negative` classes but to **map** classification **predictions to film rating**. A naive way to do that is multiply the probability of `positive` class by `9` and add `1`. But it doesn't work correctly because of the distribution of predictions. Let's get closer look on it.

Histogram below represents distribution of film ratings at imdb (according to [this dataset](https://raw.githubusercontent.com/miptgirl/kinopoisk_data/master/kp_all_movies.csv)). Our dataset consists of `positive` and `negative` class, or only reviews with rating `> 7` and `< 4` accordingly. `Neutral` class is not represented in dataset.

![D-1](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/distribution-1.png)

When we mapping our `positive`-class probability to the rating we want to create look-alike distribution in production. And re-scaling our probability to 1-10 range is not the way to get it. Below is histogram with distribution of model predictions. Colors matching the distribution of film ratings to the distribution of predictions.

![D-2](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/distribution-2.png)

After rescaling this distribution in parts we get distribution showed below.

![D-3](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/distribution-3.png)

Last we need is to create map from this distribution to initial film rating distribution from imdb, e.g. by modified  `Inverse Transform Sampling` method (Implementation is stored here: `./modules/dist_map.py`), using small amout of bins for smoother look. `x` here represents predictions axis (multiplied by 10), and `y` - rating axis. As we can see, that is exactly what we looking for: very low probability to get too high and too low ratings (like in real distribution!)

![Map](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/map.png)

---

## Testing Rating model

For testing rating model i have scraped `700` reviews from imdb with their scores. This small dataset is stored in `./csv/review_ratings.csv`. Plot below shows predictions mapped to rating in two ways: simple multiplying by `9` and adding `1`, or `rescaling` and using function described before, or `mapping`. Red line shows ground truth labels. As you can see, simple rescaling works worse.

You can see dataset gathering and evaluating this metrics in `./Movie Rating Test.ipynb`. 

|Method   |MSE_1 |MSE_2|MSE_3|MSE_4|MSE_5|MSE_6|MSE_7|MSE_8|MSE_9|MSE_10|Mean  |
|---------|------|-----|-----|-----|-----|-----|-----|-----|-----|------|------|
|Rescaling|4.771 |5.009|5.871|6.201|9.084|7.536|6.369|3.731|1.931|4.072 | 5.443|
|Mapping  |12.504|8.75 |5.421|2.786|2.565|1.603|1.328|1.263|2.697|6.537 | 4.423|


![Rating test](https://github.com/Datasciensyash/ReviewClassifier/raw/master/images/dist_compare.png)

## Deploy and API

Model has been [deployed](https://filmreviewclassifier.herokuapp.com/model_handler/?input=None) on Django at heroku. For closer look you can see `./api/` folder. 

Example of use:
```python
import requests
url = 'https://filmreviewclassifier.herokuapp.com/model_handler/'
review = 'This film is so boring, i just fall asleep'
requests.get(url, {'input': review})

>> {"Predictions": [{"Class": -1, "Description": "Negative", "Rating": 1.3, "Rating_rounded": 1}]}
```
Fields in response:
- `Class`: Class of review.
  - `-1` is negative.
  - `0` is neutral.
  - `1` is positive.
- `Description`: description of class.
- `Rating`: Rating of film based on rating model.
- `Rating_rounded`: `Rating`, but rounded.

---

## Pretrained models

Pretrained models are stored in `./models/` directory.
- `model.pkl`: Model of `LogisticRegression`
- `vectorizer.pkl`: Model of `TfidfVectorizer`.
- `rating.pkl`: Model of `DistributionMap` (`./modules/dist_map.py`)

---

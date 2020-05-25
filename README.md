# ReviewClassifier

Model and api to classifiy film reviews on 10 point scale.

---

## Dataset

#### Dataset source
For training and evaluating model was chosen [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), gathered at [IMDB](https://www.imdb.com/). Dataset contains `50,000` user film reviews with positive or negative label. `25,000` in the test and training part.

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
Dataset was originally stored in `50,000` `.txt` files, but during the data preparation process was created pandas `DataFrame`'s with train and test samples. Also, i have performed the deletion of not-unique samples from dataset, which reduced size of train and test `24,904` and `24,801` respectively.
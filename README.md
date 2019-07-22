# Movie genre classifier

This simple command-line Python application infers the genre of a movie given its title and a short plot description, by employing basic natural language processing and machine learning approaches.

## Installation

Prerequisites for installing this app are [Python 3.7](https://www.python.org/downloads/) and the [pip](https://pypi.org/project/pip/) package installer.

The essential dependencies for running the app are collected in `requirements.txt`, and they include
  - [nltk](https://www.nltk.org) for natural language processing tools such as stemming and stopwords,
  - [joblib](https://joblib.readthedocs.io/en/latest/) for ensuring model persistence,
  - [scikit-learn](https://scikit-learn.org/stable/index.html) for machine learning classification tools.

Full list of dependencies necessary for development purposes (e.g., for retraining the model) is collected in `requirements_dev.txt`, which also include
  - [pandas](https://pandas.pydata.org/) for dealing with datasets
  - [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org) for data visualization
  - [jupyter](https://jupyter.org/)
  - [numpy](https://numpy.org/)

To get the app and install the dependencies, navigate to a convenient directory and run the following commands from the terminal:

```shell
$ git clone https://github.com/beingkk/movie-genre-classifier
$ cd movie-genre-classifier
$ python3 -m venv movie_genre_classifier
$ source movie_genre_classifier/bin/activate
$ pip3 install -r requirements.txt
```


## Usage
To start predicting movie genres, try the following:

```shell
$ python3 movie_classifier.py --title "The Matrix" --description "A programmer is brought back to reason and reality when learning he was living in a program created by gigantic machines which make human birth artificial. In order to set humanity free, Neo will have to face many enemies by using technologies and self-trust."
{
    "title": "The Matrix",
    "description": "A programmer is brought back to reason and reality when learning he was living in a program created by gigantic machines which make human birth artificial. In order to set humanity free, Neo will have to face many enemies by using technologies and self-trust.",
    "genre": "Science Fiction, Action"
}
```

Note that the app outputs several genres in order to provide the user with a better insight about the nature of the movie under question. The genres are displayed in a descending order with respect to the confidence (probability) that the classification algorithm has assigned. For example, in the case shown above, the algorithm regards "The Matrix" more as Science Fiction.

### Running with Docker

It is also possible to use a [Docker](https://cloud.docker.com/u/karliskanders/repository/docker/karliskanders/movie-classifier/) image to run the app. In this case, the syntax is slightly different (see also `Dockerfile` for implementation details):

```shell
$ docker run -e title="The Matrix" -e description="A programmer is brought back to reason and reality when learning he was living in a program created by gigantic machines which make human birth artificial. In order to set humanity free, Neo will have to face many enemies by using technologies and self-trust" karliskanders/movie-classifier
{
    "title": "The Matrix",
    "description": "A programmer is brought back to reason and reality when learning he was living in a program created by gigantic machines which make human birth artificial. In order to set humanity free, Neo will have to face many enemies by using technologies and self-trust",
    "genre": "Science Fiction, Action"
}
```


## Implementation of the classification algorithm

The full step by step implementation of the data preparation and genre classification algorithm can be found in the jupyter notebook `Training.ipynb`. This notebook can be used to either reproduce or iterate and improve upon these results.

### Data preprocessing

In brief, the dataset used for the training was "The Movies Dataset" from [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv). To aid in the reproduction of this work, the corresponding data `the-movies-dataset/movies_metadata.csv` is also included in this repository.

To obtain more robust and meaningful features, the textual data goes through several preprocessing steps. The input strings to the app (title and description) are first merged together; then apostrophes, other non-alphabetical symbols and [stopwords](https://en.wikipedia.org/wiki/Stop_words) are removed, and all words are set to lower case. Finally, [stemming](https://en.wikipedia.org/wiki/Stemming) is performed (see `text_wrangling_util.py` for more details)

For example, the input text pertaining to "Toy Story" is transformed from
```
Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene...
```
to this
```
toy stori led woodi andi toy live happili room andi birthday bring buzz lightyear onto scene...
```

The processed text was vectorized using [tf-idf](https://en.wikipedia.org/wiki/Tf–idf) representation, which is often used in document classification. It provides a score for each unique word in the movie's description, by taking into account both how often it appears in the movie's description and how specific it is. This approach was chosen as a robust first attempt, but other alternatives also exist, e.g., count vectorizer.

### Model

The movie genres were estimated by using [one-versus-rest approach](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest), where for each class (i.e., movie genre) a separate classifier is trained. This method provides a confidence score for each genre, allowing to implement multi-class classification. While this approach does not take into account possible correlations between the classes that might aid in the classification task (e.g., Action movies are also likely to be Thrillers), it is computationally inexpensive compared to other, more complex methods: The computational load scales linearly as *O*(*#classes*).

Three different classifiers were very briefly tested, and the best was found to be the [Logistic regression classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), which is then also used in the app. The particular threshold value at which to make the binary decision about the assignment of the genre was also tuned to obtain the best performance (as measured by the [F1 score](https://en.wikipedia.org/wiki/F1_score), which reflects both precision and recall of the model; see the graph below)

<center>
  <img src="https://github.com/beingkk/movie-genre-classifier/blob/master/pipe_lr.png?raw=true" width=50% height=50%></center>

This particular model shown above is presently also saved as a "benchmark" `movie_genre_classifier_benchmark.joblib`. By using `Training.ipynb`, a new model can be retrained and saved over `movie_genre_classifier.joblib`, and the new model can then be compared with the "benchmark" by running `tests.py`.

### Final remarks

The modelling approach taken here has been rather simple, and it is foreseeable that the performance in terms of F1 score could be improved by tuning the parameters, or employing classification approaches that take into account correlations between genres. Another important issue is that the dataset is imbalanced (i.e., some genres like Drama and Comedy are much more common than, e.g., War or Western). This in principle should be somehow addressed, e.g, by undersampling the overrepresented genres or synthesizing more examples of the underrepresented genres (if the underlying data structure permits that).

Nonetheless, the present model already exhibits quite nice behavior. For example, different synopses of the same show yield congruent inferences:

```shell
$ python3 movie_classifier.py --title "Chilling Adventures of Sabrina" --description "Reimagines the origin and adventures of Sabrina: the Teenage Witch as a dark coming-of-age story that traffics in horror, the occult and, of course, witchcraft. Tonally in the vein of Rosemary's Baby and The Exorcist, this adaptation finds Sabrina Spellman wrestling to reconcile her dual nature - half-witch, half-mortal - while standing against the evil forces that threaten her, her family and the daylight world humans inhabit."
{
    "title": "Chilling Adventures of Sabrina",
    "description": "Reimagines the origin and adventures of Sabrina: the Teenage Witch as a dark coming-of-age story that traffics in horror, the occult and, of course, witchcraft. Tonally in the vein of Rosemary's Baby and The Exorcist, this adaptation finds Sabrina Spellman wrestling to reconcile her dual nature - half-witch, half-mortal - while standing against the evil forces that threaten her, her family and the daylight world humans inhabit.",
    "genre": "Horror, Fantasy, Comedy"
}
```

```shell
$ python3 movie_classifier.py --title "Chilling Adventures of Sabrina" --description "A dark drama about a teen girl (Kiernan Shipka) with magical powers. Existing in the same world as Riverdale and classic Archie comic characters, it's an edgy retelling of the young witch's story. Violence includes deaths, stabbings with spurting blood, images of witches hanging from a tree, a character making a joke about having human flesh for dinner while standing over the body of a recently murdered teen, and more."
{
    "title": "Chilling Adventures of Sabrina",
    "description": "A dark drama about a teen girl (Kiernan Shipka) with magical powers. Existing in the same world as Riverdale and classic Archie comic characters, it's an edgy retelling of the young witch's story. Violence includes deaths, stabbings with spurting blood, images of witches hanging from a tree, a character making a joke about having human flesh for dinner while standing over the body of a recently murdered teen, and more.",
    "genre": "Fantasy, Horror"
}
```

Similarly, also for two different synopses of Stanley Kubrick's "Paths of Glory". Note that in both cases the model proposes new genres in addition to the ones that were provided in the original dataset (War and Drama)

```shell
$ python3 movie_classifier.py --title "Paths of Glory" --description "During World War I, commanding officer General Broulard (Adolphe Menjou) orders his subordinate, General Mireau (George Macready), to attack a German trench position, offering a promotion as an incentive. Though the mission is foolhardy to the point of suicide, Mireau commands his own subordinate, Colonel Dax (Kirk Douglas), to plan the attack. When it ends in disaster, General Mireau demands the court-martial of three random soldiers in order to save face."
{
    "title": "Paths of Glory",
    "description": "During World War I, commanding officer General Broulard (Adolphe Menjou) orders his subordinate, General Mireau (George Macready), to attack a German trench position, offering a promotion as an incentive. Though the mission is foolhardy to the point of suicide, Mireau commands his own subordinate, Colonel Dax (Kirk Douglas), to plan the attack. When it ends in disaster, General Mireau demands the court-martial of three random soldiers in order to save face.",
    "genre": "War, Action, Drama"
}
```

```shell
$ python3 movie_classifier.py --title "Paths of Glory" --description "In \"Paths of Glory\" war is viewed in terms of power. This film about a true episode in World War I combines the idea that class differences are more important than national differences with the cannon-fodder theory of war, the theory that soldiers are merely pawns in the hands of generals who play at war is if it were a game of chess."
{
    "title": "Paths of Glory",
    "description": "In \"Paths of Glory\" war is viewed in terms of power. This film about a true episode in World War I combines the idea that class differences are more important than national differences with the cannon-fodder theory of war, the theory that soldiers are merely pawns in the hands of generals who play at war is if it were a game of chess.",
    "genre": "War, Drama, History"
}
```

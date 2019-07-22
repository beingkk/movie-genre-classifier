# Movie genre classifier

This simple command-line Python application infers the genre of a movie given its title and a short plot description, by employing basic natural language processing and machine learning approaches.

## Installation

Prerequisites for installing this app are [Python 3.7](https://www.python.org/downloads/) and the [pip](https://pypi.org/project/pip/) package installer.

The essential dependencies for running the app are collected in `requirements.txt`, and they include
  - [nltk](https://www.nltk.org) for natural language processing tools such as stemming and stopwords,
  - [joblib](https://joblib.readthedocs.io/en/latest/) for model persistence,
  - [scikit-learn](https://scikit-learn.org/stable/index.html) for machine learning classification tools.

Full list of dependencies necessary for development purposes (e.g., for retraining the model) are collected in `requirements_dev.txt`, which include
  - [pandas](https://pandas.pydata.org/) for dealing with datasets
  - [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org) for data visualization
  - [jupyter](https://jupyter.org/)
  - [numpy](https://numpy.org/)

To get the app and install the dependencies, navigate to a convenient directory and run the following commands from the terminal:

```shell
$ python3 -m venv movie_genre_classifier
$ source movie_genre_classifier/bin/activate

$ git clone https://github.com/beingkk/movie-genre-classifier
$ pip3 install -r requirements.txt
```

----

## Usage
To start predicting movie genres, try the following:

```shell
$ python3 movie_classifier.py --title "The Matrix" --description "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
{
    "title": "The Matrix",
    "description": "A programmer is brought back to reason and reality when learning he was living in a program created by gigantic machines which make human birth artificial. In order to set humanity free, Neo will have to face many enemies by using technologies and self-trust.",
    "genre": "Science Fiction, Action"
}
```

Note that the app outputs several genres in order to provide the user with a better insight about the nature of the movie under question. 

### Running with Docker

It is also possible to use a [Docker](cloud.docker.com/u/karliskanders/repository/docker/karliskanders/movie-classifier/) image to run the app. In this case, the syntax is slightly different (see also `Dockerfile` for implementation details):

```shell
$ docker run -e title="The Matrix" -e description="A programmer is brought back to reason and reality when learning he was living in a program created by gigantic machines which make human birth artificial. In order to set humanity free, Neo will have to face many enemies by using technologies and self-trust" karliskanders/movie-classifier
{
    "title": "The Matrix",
    "description": "A programmer is brought back to reason and reality when learning he was living in a program created by gigantic machines which make human birth artificial. In order to set humanity free, Neo will have to face many enemies by using technologies and self-trust",
    "genre": "Science Fiction, Action"
}
```

----

## Implementation of the classification algorithm

The full implementation details of the algorithm can be found in the jupyter notebook `Training.ipynb`


##

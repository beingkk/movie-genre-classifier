#!/usr/bin/env python3
"""
Movie genre classifier
Created in July 2019
Author: Karlis Kanders
"""

import argparse
import joblib
import json

import text_wrangling_util

if __name__ == "__main__":

    """ Parse and process input """

    parser = argparse.ArgumentParser(description="Movie genre classifier predicts a movie's genre from its title and plot description. " +
    "Try: python3 movie_classifier.py --title \"The Matrix\" --description \"In the distant future, a hacker Neo " +
    " realizes he is part of a simulation called Matrix and he needs to save the surviving humans by fighting bad programs.\"")

    parser.add_argument("-t", "--title", help="English title of the movie", type=str)
    parser.add_argument("-d", "--description", help="Short description of the plot", type=str)

    args = parser.parse_args()

    if not args.title or not args.description:
        print("Please provide both the title and the description of the movie.")
        exit()

    if args.title == "" or args.description == "":
        print("Title and description must not be empty.")
        exit()

    input_text = text_wrangling_util.prepare_input_text(args.title, args.description)

    if input_text == [""]:
        print("Title and description must contain letters from the English alphabet...")
        exit()


    """ Load model and perform inference """

    try:
        model = joblib.load('movie_genre_classifier.joblib')
    except FileNotFoundError:
        print("Model specification \"movie_genre_classifier.joblib\" not found.")
        print("Download the model or retrain the network using Training.ipynb notebook.")
        exit()

    Y_pred_prob = model["pipeline"].predict_proba(input_text)[0]

    # Display genres in the order of descending probabilities
    genres = [genre for (prob, genre) in sorted(zip(Y_pred_prob, model["genres"]))[::-1] if prob > model["threshold"]]

    output_dict = {"title":args.title, "description":args.description, "genre": ", ".join(genres)}
    print(json.dumps(output_dict, indent=4))

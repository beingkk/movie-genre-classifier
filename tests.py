import unittest
import joblib
import text_wrangling_util

class TestMovieClassifier(unittest.TestCase):

    def test_model_performance_better_or_equal_to_benchmark(self):
        model = joblib.load("movie_genre_classifier.joblib")
        model_benchmark = joblib.load("movie_genre_classifier_benchmark.joblib")
        self.assertGreaterEqual(model["score"], model_benchmark["score"])

    def test_model_file_has_correct_contents(self):
        model = joblib.load("movie_genre_classifier.joblib")
        expected_keys = ['pipeline', 'threshold', 'genres', 'score']
        self.assertEqual(list(model.keys()), expected_keys)

    def test_input_wrangled_correctly_with_one_movie(self):
        title = "Toy Story"
        description = "Movie made in 1995 by the Pixar studios about the life of toys"
        input_text = text_wrangling_util.prepare_input_text(title, description)
        expected_text = ["toy stori movi made pixar studio life toy"]
        self.assertEqual(type(expected_text),list)
        self.assertEqual(input_text, expected_text)

    def test_input_wrangled_correctly_with_two_movies(self):
        titles = ["Toy Story","Avatar"]
        descriptions = ["Movie made in 1995 by the Pixar studios about the life of toys",
        "Humans arrive on a beatiful planet and meet their mysterious blue inhabitants"]
        input_text = text_wrangling_util.prepare_input_text(title=titles, description=descriptions)
        expected_text = ["toy stori movi made pixar studio life toy",
        "avatar human arriv beati planet meet mysteri blue inhabit"]
        self.assertEqual(type(expected_text),list)
        self.assertEqual(input_text, expected_text)

if __name__ == '__main__':
    unittest.main(verbosity=2)


import numpy as np
from surprise import accuracy


def fit_surprise_algorithm(algo, trainset):
    """
    Fit the algorithm to the Surprise training set. 
    Compute and print MAE and RMSE on the training set.
    """
    algo.fit(trainset)

    # Compute training error (MAE and RMSE)
    predictions_train = algo.test(trainset.build_testset())
    accuracy.mae(predictions_train, verbose=True)
    accuracy.rmse(predictions_train, verbose=True)

    return algo


def evaluate_performance(algo, testset):
    """
    Return the rating predictions for a Surprise test set. 
    Compute and print MAE and RMSE on the test set.
    """
    
    # Predict ratings
    predictions_test = algo.test(testset)

    # Compute test error (MAE and RMSE)
    accuracy.mae(predictions_test, verbose=True)
    accuracy.rmse(predictions_test, verbose=True)

    return predictions_test


def get_top_n_recommendations(predictions, n=10):
    """
    Return the top N recommendations (book ids) from a list of predictions 
    made for the user.

    Parameters:

    predictions : list of Prediction objects 
        Predictions, as returned by the test method of an algorithm.
    n : int, default 10 
        The number of recommendations to output for the user.

    Returns:

    list of int
        The top N recommended book ids
    """

    # Append tuples with the book id and predicted rating to a list
    all_predictions = []
    for user_id, book_id, true_rating, rating_est, details in predictions:
        all_predictions.append((book_id, rating_est))

    # Sort the predictions and retrieve the n highest ones
    all_predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_book_ids = [book_id for book_id, rating_est in all_predictions[:n]]

    return top_n_book_ids


def get_book_details(book_id, books_df):
    """
    Return the author, title and year of original publication for 
    a given book id.
    """
    author = books_df.loc[book_id, 'author']
    title = books_df.loc[book_id, 'title']
    year = books_df.loc[book_id, 'original_publication_year']
    return author, title, year


def print_recommendation(book_details):
    """
    Print book details.

    Parameters:
    book_details : tuple (author, title, year)
    """
    author, title, year = book_details

    if np.isnan(year):
        print(f'{author}. {title}.\n')
    else:
        print(f'{author}. {title}, {int(year)}\n')

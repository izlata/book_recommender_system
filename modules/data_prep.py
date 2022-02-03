import pandas as pd
from surprise import Dataset
from surprise import Reader


def load_data_surprise(file_path, min_rating, max_rating):
    """
    Load the data from a csv file containing user id, item id and ratings 
    into a Surprise Dataset.
    """

    # Read data into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Define a Reader object for Surprise to be able to parse the DataFrame 
    reader = Reader(rating_scale=(min_rating, max_rating))

    # Load a dataset 
    # Columns in df must correspond to user id, item id and ratings (in this order)
    data = Dataset.load_from_df(df, reader)
    return data


def build_trainset_surprise(dataset):
    """
    Build a training set from a Surprise Dataset so that 
    it can be used for model training.
    """
    trainset = dataset.build_full_trainset()
    return trainset


def build_testset_surprise(dataset):
    """
    Build a test set from a Surprise Dataset so that it can be used 
    for making predictions and model evaluation.
    """
    testset = dataset.build_full_trainset().build_testset()
    return testset


def load_ratings(file_path):
    """ Read rating data from a csv file into a pandas DataFrame. """

    ratings_df = pd.read_csv(file_path)
    return ratings_df


def load_book_details(file_path):
    """ Read book details from a csv file into a pandas DataFrame. """

    books_df = pd.read_csv(file_path, index_col='book_id')
    return books_df


def preprocess_user_data(user_id, ratings_df):
    """
    Build a Surprise test set with book ids of the books 
    that have not been rated by a given user. 
    """
    rated_books = set(ratings_df.loc[ratings_df['user_id'] == user_id, 'book_id'])
    all_books = set(ratings_df['book_id'])
    books_unknown_rating = sorted(all_books - rated_books)

    # Create a test set for the user so that it can be used to predict unknown ratings
    user_testset = []
    for i in books_unknown_rating:
        # Some value should be passed as a true rating 
        # Here, 3.92 is used - mean rating (this value will not affect the prediction)
        user_testset.append((user_id, i, 3.92))
    
    return user_testset

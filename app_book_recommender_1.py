import streamlit as st
import pandas as pd
import numpy as np
from surprise import dump


st.set_page_config(page_title='Book Recommendations', 
                   layout='centered', 
                   initial_sidebar_state='auto')

with st.sidebar:
    st.title('Book Recommender App')
    st.markdown('This app provides personalized book recommendations for users based on their previous ratings.')
    st.write('')

tab = st.sidebar.radio(
    label='Tabs',
    options=['About the project', 'Get recommendations'], 
    index=1, 
    key='tabs'
)


if tab == 'Get recommendations':
    
    st.title('Get Recommendations')

    st.image('images/littleprince_wikipedia.jpe', width=300)

    st.write('')

    st.markdown(
        """
        **Please enter your user ID to find 10 books recommended for you.** 
        Possible inputs are from 1 to 53424.
        """
    )


    # FUNCTIONS

    @st.cache(show_spinner=False)
    def load_csv_data(file_path, index_col=None):
        """ Read a csv file into a pandas DataFrame. """
        df = pd.read_csv(file_path, index_col=index_col)
        return df


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
        Return a string with the author, title and year of original publication 
        (if available) for a given book id.
        """
        author = books_df.loc[book_id, 'author']
        title = books_df.loc[book_id, 'title']
        year = books_df.loc[book_id, 'original_publication_year']
        
        return author, title, year


    # USER INPUT

    # Create an input field: 
    # the user should enter user ID to get personalized recommendations
    user_input = st.text_input('Input your user ID (e.g. 28001, 654)')


    # LOAD DATA

    # Load rating data
    path_ratings = 'data/ratings_train.csv'
    ratings = load_csv_data(path_ratings)
    unique_users = ratings['user_id'].unique()  # unique user IDs

    # Load book details (author, title and year of original publication)
    path_books = 'data/book_details.csv'
    index_col_books = 'book_id'
    books = load_csv_data(path_books, index_col=index_col_books)


    # CHECK THE INPUT

    # User IDs in the dataset - int values from 1 to 53424
    if user_input:
        try:
            assert user_input[0] != '0'  # first digit cannot be 0 
            user_input_int = int(user_input)  # convert string to int
            assert user_input_int in unique_users  # check if this user ID exists
        except:
            st.info('There is no profile with this user ID')
            user_input_int = None
    else:
        user_input_int = None


    # RECOMMENDATIONS (Item-based collaborative filtering)

    if user_input_int:
        
        recommendation_state = st.text('Choosing books...')

        # Load the k-NN Baseline model (trained using the Surprise library).
        # The model is loaded once during a user session, then stored and 
        # accessed as a Session State variable across app reruns.

        if 'knn_model' not in st.session_state:
            path_knn = 'models/knn_baseline_items'
            _, knn = dump.load(path_knn)
            st.session_state['knn_model'] = knn

        # Preprocess the user's data for making predictions 
        user_testset = preprocess_user_data(user_input_int, ratings)

        # Predict unknown ratings for the user
        user_predictions = st.session_state['knn_model'].test(user_testset)

        # Get top N recommended books (book ids)
        top_n_book_ids = get_top_n_recommendations(user_predictions, n=10)

        # Get book details for book ids
        recommendations = []

        for i, book_id in enumerate(top_n_book_ids):

            author, title, year = get_book_details(book_id, books)

            if np.isnan(year):  # A few books do not have the year info
                recommendations.append(f'{i+1}. **{author}.** {title}.\n')
            else:
                recommendations.append(f'{i+1}. **{author}.** {title}, {int(year)}\n')
        
        recommendations_string = ''.join(recommendations)  # a string with 10 books

        recommendation_state.empty()  # the text 'Choosing books...' will disappear

        # Display recommendations
        st.subheader('Books you may like:')
        st.markdown(recommendations_string)


if tab == 'About the project':

    st.title('Book Recommendations')

    st.image('images/alfons-morales-YLSwjSy7stw-unsplash-crop1.jpg', width=None)
    
    st.markdown(
        """
        Zlata Izvalava's Project, 2022 
        ([GitHub repository](https://github.com/izlata/book_recommender_system))
        
        Data Science Bootcamp at Lighthouse Labs
        """
    )
    
    st.header('Introduction')

    st.markdown(
        """
        - Many companies use online recommender systems on their websites to 
        improve customer experience and increase company profits. 
        These techniques help better understand users and their tastes and 
        produce personalized recommendations. 
        - The goal of my project is to build a recommender system that suggests 
        relevant books based on the person's interests. 
        - This web app takes a user ID as input and displays a list of 10 recommended books.
        """
    )

    st.header('Dataset')

    st.markdown(
        """
        I used the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset 
        created by Zygmunt Zając.
        
        Data includes: 
        - 10,000 books (different editions of the book have the same book ID)
        - 53,424 users
        - 6M ratings (ratings are whole numbers from 1 to 5)
        """
    )

    st.header('Approach')

    st.markdown(
        """
        1. Data Exploration
        2. Modeling - collaborative filtering using [Surprise](http://surpriselib.com/) library
            - Compare different Surprise algorithms on the training set (cross-validation)
            - Evaluate shortlisted models on the test set and choose the final model
        3. Create and deploy a Streamlit web app that recommends 10 books for the user 
        """
    )

    st.header('Data Exploration')

    st.subheader('Ratings')

    st.markdown(
        """
        The mean rating is about 3.92. Higher ratings are more common. Plot the distribution of ratings:
        """
    )

    st.image('images/ratings_bar_chart.png')

    st.markdown(
        """
        Each user rated at least 15 books, and on average users gave about 95 ratings.
        
        Average number of ratings per book is 508. 
        """
    )

    st.subheader('Books')

    st.markdown(
        """
        - There are books from ancient times (The Epic of Gilgamesh, c. 1750 B.C.) and up to 2017.
        - Authors with the highest number of books in this dataset: Nora Roberts, James Patterson, 
        Stephen King, Dean Koontz, Terry Pratchett, Agatha Christie.
        """
    )

    st.markdown(
        """
        #### Popular Books: 
        These books have many ratings and high average ratings. 
        - **J.K. Rowling**: Harry Potter series
        - **Kathryn Stockett**: The Help
        - **George R.R. Martin**: A Game of Thrones
        - **Markus Zusak**: The Book Thief
        - **J.R.R. Tolkien**: The Fellowship of the Ring, The Hobbit
        - **Suzanne Collins**: The Hunger Games
        - **Shel Silverstein**: Where the Sidewalk Ends
        - **Khaled Hosseini**: The Kite Runner
        - **John Green**: The Fault in Our Stars
        - **Harper Lee**: To Kill a Mockingbird
        """
    )

    st.header('Modeling: Collaborative Filtering')

    st.markdown(
        """
        1. [**Normal Predictor**](https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) (random predictions based on rating distribution) - **as a baseline model** 
            - Test MAE = **1.05**
        2. [**Slope One**](https://surprise.readthedocs.io/en/stable/slope_one.html)
            - Test MAE = 0.66
        3. [**SVD++**](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp) 
            - Test MAE = 0.63
        4. [**k-NN Baseline, item-based**](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline) - **the best model**
            - Test MAE = **0.60**
            - The Mean Absolute Error on the test set has been reduced by 42% compared to the Normal Predictor (from 1.05 to 0.60)
            - On average, the rating predictions by k-NN model are off by 15% (mean rating = 3.92)

        ##### The k-NN Baseline model is used in this app to predict unknown ratings for the user. Then the books with the highest predicted ratings are included in the recommendation list.
        """
    )

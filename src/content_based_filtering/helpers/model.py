# from src.content_based_filtering.helpers.make_dataset import User,
from tqdm import tqdm
import pandas as pd
import numpy as np

class Model:
    """
    a wrapper for the recommendation model
    :users_db_object: (UsersDB) the users database
    :users_db: (dict) the mapping between the ids and the User objects
    :movies_db: (MoviesDB) the movies database
    """
    def __init__(self, UsersDB, MoviesDB):
        self.users_db_object = UsersDB
        self.users_db = self.users_db_object.db
        self.movies_db = MoviesDB

    def predict_content_based(self):
        """
        returns the most similar movies for every user

        Returns: (dict) prediction

        """
        prediction = {}

        for user_id in tqdm(self.users_db_object.users):
            prediction[user_id] = self.users_db[user_id].get_recommendations(MoviesDB=self.movies_db)

        return prediction

    def predict_content_based_one_user(self, user_id):
        """
        the content-based prediction
        Args:
           (int) user_id: the id of the user we wish to predict the content for

        Returns:

        """
        return self.users_db[user_id].get_recommendations(MoviesDB=self.movies_db)

    @staticmethod
    def predict_similar_users_one_user(user, similarity_matrix, UsersDB, top):
        """
        returns the recommended movies based on what the similar users have seen
        Args:
            (User) user: a user object representing the user
            (array) similarity_matrix:
            (UserDB) UsersDB: the users database object
            (int) top: the length of the prediction

        Returns:
            (DataFrame) movie_pool
        """

        similar_users = user.get_similar_users(similarity_matrix=similarity_matrix)
        similar_users = list(zip(*similar_users))[0]  # only take the user_ids
        users = [UsersDB.db[user_id] for user_id in similar_users]
        users_ratings = [user.ratings.sort_values(by='rating', ascending=False).head(5) for user in
                         users]  # get the top movies from each user

        movie_pool = pd.concat(users_ratings).drop_duplicates('movie_id').sort_values(by='movie_id')

        return movie_pool.head(5)

    @staticmethod
    def score(users_similarity_matrix, content_based_prediction, user, UsersDB):
        """
        computes the score of the prediction based on the ratings on the movies
        seen by the other users
        Args:
            (array) users_similarity_matrix: the similarity matrix between the users
            (DataFrame) content_based_prediction: the content-based prediction
            (User) user: the user we wish to evaluate the quality of the prediction on
            (UserDB) UsersDB: the users database

        Returns:
            (float) score
        """
        prediction = content_based_prediction
        similar_users = user.get_similar_users(similarity_matrix=users_similarity_matrix)
        users_id = list(zip(*similar_users))[0]
        score = 0

        # computes the movies seen by all the similar users
        seen_movies_dict = dict()
        movie_list = []
        for us_id in users_id:
            movie_list += list(UsersDB.db[us_id].seen_movies)

        cleansed_movies_list = list(dict.fromkeys(movie_list))  # removes duplicates

        for movie_id in cleansed_movies_list:
            seen_movies_dict[movie_id] = []

        for movie_id in cleansed_movies_list:
            for us_id in users_id:
                usr_ratings = UsersDB.db[us_id].ratings

                if len(usr_ratings.loc[
                           usr_ratings.movie_id == movie_id].rating.values) > 0:  # checks if the movie was seen by the user
                    seen_movies_dict[movie_id].append(
                        usr_ratings.loc[usr_ratings.movie_id == movie_id].rating.values[0])

        for movie_id, ratings_list in seen_movies_dict.items():
            seen_movies_dict[movie_id] = np.mean(np.array(ratings_list))  # computes the mean of all the users

        for movie_id in prediction.movie_id.values:
            if movie_id in seen_movies_dict:
                score += seen_movies_dict[movie_id]

        return score

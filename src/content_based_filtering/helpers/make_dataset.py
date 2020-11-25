import pandas as pd
from tqdm import tqdm


class Movies:
    """
    base movie dataset class that contains all the movies

    :movies_dataset: (DataFrame) the df representing the movies.
    It has 3 columns describing the basic metadata of the movie (title, year, movie_id)
    and a certain number of columns describing its genre in a multi-hot encoded fashion.
    :genre_cols: (list<str>) the genre names of the movies
    :similarity_matrix: (numpy array (n_movies x n_movies)) a matrix that denotes the similarity between movies,
    deriving from the similarity metric defined. (defaulting scalar product between the genre multi-hot encoded vectors
    of each movies)

    """

    def __init__(self, movies_dataset):
        # preprocessing
        tmp = pd.DataFrame([i for i in range(movies_dataset.movie_id.values[-1] + 1)])
        tmp.columns = ['movie_id']
        result = pd.merge(movies_dataset, tmp, how='outer', on=['movie_id'])

        self.movies_dataset = result.sort_values(by='movie_id').fillna(0).reset_index().drop(columns=['index'])
        self.genre_cols = [name for name in movies_dataset.columns if name not in ('title', 'year', 'movie_id')]
        self.similarity_matrix = self.movies_dataset[self.genre_cols].values.dot(
            self.movies_dataset[self.genre_cols].values.T)  # (n_movies, n_movies)

    def get_movie_id(self, title, year=None):
        """

        Args:
            (str) title: the title of the chosen movie
            (int) year: the year of the movie

        Returns:
            (int) id : the id of the movie whose title was provided

        """
        res = self.movies_dataset[self.movies_dataset['title'] == title]
        if year:
            res = res[res['year'] == year]

        if len(res) > 1:
            print("Ambiguous: found")
            print(f"{res['title']} {res['year']}")
        elif len(res) == 0:
            print('not found')
        else:
            return res.movie_id.values[0]

    def get_movie_name(self, movie_id):
        """

        Args:
            (int) movie_id: the id of the movie

        Returns:
            (str) name : the name of the movie

        """
        return self.movies_dataset.loc[self.movies_dataset.movie_id == movie_id].title.values[0]

    def get_movie_year(self, movie_id):
        """

        Args:
            (int) movie_id: the id of the movie

        Returns:
            (int) year : the movie year
        """

        return self.movies_dataset.loc[self.movies_dataset.movie_id == movie_id].year.values[0]

    def get_most_similar_movies(self, movie_name, year=None, top=10):
        """

        Args:
            (str) movie_name: the name of the movie
            (int) year: the year of the movie
            (int) top: the desired number of similar movies to return

        Returns:
            (list) best_movies : the movies the most similar to the provided one

        """

        index_movie = self.get_movie_id(movie_name, year)
        best = self.similarity_matrix[index_movie].argsort()[::-1]
        return [(ind, self.get_movie_name(ind), self.similarity_matrix[index_movie, ind]) for ind
                in
                best[:top] if
                ind != index_movie]


class Ratings:
    """
    the ratings given by all the users to each movies he's/she's seen
    :ratings: (DataFrame)
    :users_id: (list<int>) id of the users in the ratings DB
    :movies_id: (list<int>) id of the movies in the ratings DB
    """

    def __init__(self, RatingsDB):
        self.ratings = RatingsDB
        self.users_id = self.ratings.user_id.unique()
        self.movies_id = self.ratings.movie_id.unique()

    def get_user_ratings(self, user_id):
        """

        Args:
            (int) user_id: the id of the user whose ratings we wish to fetch

        Returns:
            (DataFrame) ratings_user: the ratings of the wanted user
        """
        return self.ratings.loc[self.ratings.user_id == user_id]


class User:
    """
    the user we wish to predict recommendations for
    :user_id: (int) the user's id
    :user: (DataFrame) the characteristics of the user
    :ratings_db: (Ratings) feedback of the user on the movies he's/she's seen
    :ratings: (DataFrame) the ratings given by the user
    :seen_movies: (list<int>) the ids of the movies the user has seen
    """

    def __init__(self, user_id, users_dataset, RatingsDB):
        self.user_id = user_id
        self.user = users_dataset.loc[users_dataset.user_id == user_id]
        self.ratings_db = RatingsDB
        self.ratings = RatingsDB.get_user_ratings(user_id)
        self.seen_movies = self.ratings.movie_id.values

    def get_recommendations(self, MoviesDB):
        """
        returns the recommendations for the content-based prediction
        Args:
            (DataFrame) movies_dataset: the movies DB

        Returns:
            (DataFrame) recommendations for the user
        """
        top_movies = self.ratings.sort_values(by='rating', ascending=False).head(3)['movie_id']
        index = ['movie_id', 'title', 'similarity']

        most_similars = []
        for top_movie in top_movies:
            most_similars += MoviesDB.get_most_similar_movies(MoviesDB.get_movie_name(top_movie),
                                                              MoviesDB.get_movie_year(top_movie))

        return pd.DataFrame(most_similars, columns=index).drop_duplicates().sort_values(by='similarity',
                                                                                        ascending=False).head(5)

    def get_encoded_ratings(self, MoviesDB):
        """
        return multi hot encoded ratings for each user
        Args:
            (MoviesDB) MoviesDB: the movies DB

        Returns:
            (DataFrame) encoded_ratings: multi encoded ratings for each user

        """
        # tmp dataframe to join on the ratings one and make the multi hot encoding
        tmp = pd.DataFrame([i for i in range(MoviesDB.movies_dataset.movie_id.values[-1] + 1)])
        tmp.columns = ['movie_id']

        # merges the two df, fills the movies unseen by the user with zeros and drops the unnecessary columns
        encoded_ratings = pd.merge(self.ratings_db.get_user_ratings(self.user_id), tmp, how='outer', on=['movie_id'])
        encoded_ratings = encoded_ratings.fillna(0).sort_values(by='movie_id').reset_index().drop(columns=['index'])
        encoded_ratings = encoded_ratings.set_index('movie_id').drop(columns=['user_id'])
        encoded_ratings.columns = [self.user_id]

        return encoded_ratings

    def get_similar_users(self, similarity_matrix, top=5):
        """
        returns the most similar users
        Args:
            (array) similarity_matrix:
            (int) top: the number of users we wish to fetch

        Returns:
            (list<tuple<int, float>>) list of ids and similarity tuples
        """
        best = similarity_matrix[self.user_id].argsort()[::-1]
        return [(ind, similarity_matrix[self.user_id, ind]) for ind
                in
                best[:top] if
                ind != self.user_id]


class UserDB:
    """
    The wrapper for the whole user database
    :users_dataset: (DataFrame) users dataframe
    :users: (list<int>) list of users' ids
    :ratings_db: (RatingsDB) the ratings database
    :db: (dict) a mapping from the users' id to the user objects
    """
    def __init__(self, users, users_dataset, RatingsDB):
        self.users = users
        self.users_dataset = users_dataset
        self.ratings_db = RatingsDB
        self.db = {user_id: User(user_id, users_dataset=self.users_dataset, RatingsDB=self.ratings_db) for user_id
                   in
                   self.users}

    def get_encoded_ratings_db(self, MoviesDB):
        """
        returns the encoded ratings for the whole users database
        Args:
            MoviesDB:

        Returns:
            (DataFrame) encoded: the dataframe regrouping all the users' ratings
        """
        encoded = self.db[0].get_encoded_ratings(MoviesDB=MoviesDB)

        for user_id in tqdm(self.db.keys()):
            if user_id != 0:
                encoded = pd.merge(encoded, self.db[user_id].get_encoded_ratings(MoviesDB=MoviesDB), on='movie_id',
                                   how='inner')
        return encoded

    @staticmethod
    def get_similarity_matrix(encoded_ratings):
        """
        generates the similarity matrix
        Args:
            (DataFrame) encoded_ratings: the multi-hot encoded ratings for all the users

        Returns:
            (array) the users similarity matrix
        """
        similarity_matrix = encoded_ratings.T.values.dot(encoded_ratings.values)

        return similarity_matrix

# Movie Recommender System
Provides movie recommendations using collabirative filtering algorithms

Invoke as:

```
python3 recommender.py predict arguments
```

There is currently one possible option for _command_ and multiple options for _arguments_:

### recommender.py TrainingFile K Algorithm UserID MovieID  
This command will use simple user-based collaborative filtering to predict the rating of user userID for movie movieID, with the following parameters:

* **TrainingFile** is the training data file. (To use sample data, use u1.base).
* **K** means that the algorithm should consider only the K nearest (most similar) users to user **UserID**. Note that a value of K=0 means that there is no limit and all the users should be considered.  
* **UserID** is the user for whom we want to predict their rating for **MovieID**.
* **Algorithm** is the specific algorithm used, which can be one of the following:  
	* **average**, just computing the average rating for MovieID based on all other ratings for that movie (K is effectively set to 0 for this, regardless of user input).
	* **euclid**, when using Euclidean distance to measure user-user similarity and then use the nearest K users to UserID to predict his/her rating for MovieID (through a simple weighted average, where the similarities are the weights)  
	* **pearson**, when using Pearson Similarity to measure user-user similarity and then use the nearest K users to UserID to predict his/her rating for MovieID.
	* **cosine**, when using Cosine Similarity to measure user-user similarity and then use the nearest K users to UserID to predict his/her rating for MovieID.

##### Sample data taken from [MovieLens 100k data set](https://grouplens.org/datasets/movielens/)

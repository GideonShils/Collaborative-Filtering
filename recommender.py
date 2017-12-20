"""
A simple recommender system using different collaborative filtering algorithms
Algorithm options: Average, Euclid, Pearson, Cosine
"""

import sys
import os
from collections import Counter
import copy
import math
import numpy
import scipy.stats

# Ensure proper arguments (number and form)
if len(sys.argv) >= 6:
    command = sys.argv[1].lower()
    k_value = int(sys.argv[3])
    algorithm = sys.argv[4].lower()

    # Ensure train file exits and open if it does
    if os.path.isfile('./' + sys.argv[2]):
        train_file = open(sys.argv[2], 'r')
    else:
        print("Invalid training file. Now exiting...")
        sys.exit()

    # Ensure alrogithm choice is one of the available 4
    if algorithm != 'average' and algorithm != 'euclid' and algorithm != 'pearson' and algorithm != 'cosine':
        print("Invalid algorithm. Now exiting...")
        sys.exit()

    # Ensure  K value is greater than or equal to 0
    if k_value < 0:
        print("Invalid K value. Now exiting...")
        sys.exit()

    # Ensure proper number of arguments for each command
    if command == 'predict' or command == 'evaluate':
        if command == 'predict' and len(sys.argv) == 7:
            target_user_id = int(sys.argv[5])
            target_movie_id = int(sys.argv[6])
        elif command == 'evaluate' and len(sys.argv) == 6:
            # Ensure train file exits and open if it does
            if os.path.isfile('./' + sys.argv[5]):
                test_file = open(sys.argv[5], 'r')
            else:
                print('Invalid testing file. Now exiting...')
                sys.exit()
        else:
            print("Invalid number of arguments. Now exiting...")
    else:
        print("Invalid command. Now exiting...")
        sys.exit()

else:
    print('Invalid command. Now exiting...')
    sys.exit()

# Predict ratings using the chosen algorithm
if command == 'predict':
    # Break down the data into a dictionary for easier access
    # Format: data[user_id][movie_id] = rating
    data = {}
    for line in train_file:
        white_division = line.split('\t')
        user_id = int(white_division[0])
        movie_id = int(white_division[1])
        rating = int(white_division[2])

        # If entry doesn't exist, create it
        if user_id not in data:
            data[user_id] = {}

        # If entry exists, add to it
        data[user_id][movie_id] = rating

    # Average algorithm
    if algorithm == 'average':
        total = 0
        num_ratings = 0
        for user in data:
            # Don't include rating from target_user and only include rating if it exists
            if user != target_user_id and target_movie_id in data[user]:
                total += data[user][target_movie_id]
                num_ratings += 1

        if num_ratings != 0:
            average = total / num_ratings
        else:
            average = 3.0

        print('Command'.ljust(11) + ' = ' + command)
        print('Training'.ljust(11) + ' = ' + sys.argv[2])
        print('Algorithm'.ljust(11) + ' = ' + algorithm)
        print('K'.ljust(11) + ' = ' + str(k_value))
        print('UserID'.ljust(11) + ' = ' + str(target_user_id))
        print('MovieID'.ljust(11) + ' = ' + str(target_movie_id))
        print('Prediction'.ljust(11) + ' = ' + str(average))

    # Euclid algorithm
    elif algorithm == 'euclid':
        # Create dictionary for storing similarities to all other users
        similarities = {}

        # Iterate through users and calculate similarities
        for user in data:
            total = 0
            
            # number of movies rated by both users
            sim_count = 0
            # Iterate through movies
            for movie in data[user]:
                # Ensure the movie was rated by both users
                if movie in data[target_user_id]:
                    sim_count += 1
                    total += (data[target_user_id][movie] - data[user][movie])**2

            # Ensure sim_count is greater than 0 to weed out cases where there were no overlapping movies
            if (sim_count > 0):
                distance = total**(1/2)
                similarity = 1/(1+distance)
            else:
                similarity = 0

            similarities[user] = similarity

        #Remove self
        del similarities[target_user_id];

        if (k_value > 0):
            # Choose K closest users
            closest_users = dict(Counter(similarities).most_common(k_value))

        # Calculate weighted average
        total = 0
        weights = 0
        # If a k value was specified, only use closest users for prediction
        if k_value > 0:
            user_set = closest_users
        # If k = 0, use all users for prediction
        else:
            user_set = similarities

        # Iterate over users, summing weights and ratings
        for user in user_set:
            # Only include in prediction if target_movie was rated
            if target_movie_id in data[user]:
                # Total = sum of (rating)(similarity)
                total += similarities[user]*data[user][target_movie_id]
                # Weight = sum of similarities
                weights += similarities[user]

        # If none of the users rated the target film, default to 3.0
        if weights == 0:
            prediction = 3.0
        # Otherwise, calculate the prediction
        else:
            prediction = total / weights

        print('Command'.ljust(11) + ' = ' + command)
        print('Training'.ljust(11) + ' = ' + sys.argv[2])
        print('Algorithm'.ljust(11) + ' = ' + algorithm)
        print('K'.ljust(11) + ' = ' + str(k_value))
        print('UserID'.ljust(11) + ' = ' + str(target_user_id))
        print('MovieID'.ljust(11) + ' = ' + str(target_movie_id))
        print('Prediction'.ljust(11) + ' = ' + str(prediction))


    # Pearson algorithm
    elif algorithm == 'pearson':

        # Copy the data dict for storing normalized ratings
        norm_data = copy.deepcopy(data)

        # Normalize each rating [1:5] => [-1:1]
        for user in data:
            for movie in data[user]:
                norm_data[user][movie] = (2 * (data[user][movie] - 1) - (5 - 1)) / (5 - 1)

        # Create dictionary for storing similarities to all other users
        similarities = {}

        # Create new user vectors and calculate similarities to target vector
        for user in data:
                # Create vectors
            target_user_vector = []
            user_vector = []
            
            for movie in data[user]:
                # Only add ratings for movies both users have rated
                if movie in data[user] and movie in data[target_user_id]:
                    user_vector.append(data[user][movie])
                    target_user_vector.append(data[target_user_id][movie])

            # Calculate similarity

            # Ensure vectors have items in them
            if len(user_vector) > 2:
                # Ensure neither vector is entirely identical
                if  not all(x == user_vector[0] for x in user_vector) and not all(x == target_user_vector[0] for x in target_user_vector):
                    similarity = scipy.stats.pearsonr(target_user_vector, user_vector)

                    # Set smiilarity equal to pearson coefficient
                    similarities[user] = similarity[0]

        #Remove self
        del similarities[target_user_id];

        if (k_value > 0):
            # Choose K closest users
            closest_users = dict(Counter(similarities).most_common(k_value))

        # Calculate weighted average
        total = 0
        weights = 0

        # If a k value was specified, only use closest users for prediction
        if k_value > 0:
            user_set = closest_users
        # If k = 0, use all users for prediction
        else:
            user_set = similarities

        # Iterate over users, summing weights and ratings
        for user in user_set:
            # Only include in prediction if target_movie was rated
            if target_movie_id in norm_data[user]:
                # Total = sum of (rating)(similarity)
                total += similarities[user]*norm_data[user][target_movie_id]

                # Weight = sum of similarities
                weights += similarities[user]

        # If none of the users rated the target film, default to 3.0
        if weights == 0:
            prediction = 0
        # Otherwise, calculate the prediction
        else:
            prediction = total / weights

        prediction = (0.5 * ((prediction + 1) * (5 - 1))) + 1
        
        print('Command'.ljust(11) + ' = ' + command)
        print('Training'.ljust(11) + ' = ' + sys.argv[2])
        print('Algorithm'.ljust(11) + ' = ' + algorithm)
        print('K'.ljust(11) + ' = ' + str(k_value))
        print('UserID'.ljust(11) + ' = ' + str(target_user_id))
        print('MovieID'.ljust(11) + ' = ' + str(target_movie_id))
        print('Prediction'.ljust(11) + ' = ' + str(prediction))

    # Cosine algorithm
    else:
        # Copy the data dict for storing normalized ratings
        norm_data = copy.deepcopy(data)

        # Normalize each rating [1:5] => [-1:1]
        for user in data:
            for movie in data[user]:
                norm_data[user][movie] = (2 * (data[user][movie] - 1) - (5 - 1)) / (5 - 1)

        # Create dictionary for storing similarities to all other users
        similarities = {}

        # Create new user vectors and calculate similarities to target vector
        for user in data:
            # Create vectors
            target_user_vector = []
            user_vector = []
            
            for movie in data[user]:
                # Only add ratings for movies both users have rated
                if movie in norm_data[user] and movie in norm_data[target_user_id]:
                    user_vector.append(norm_data[user][movie])
                    target_user_vector.append(norm_data[target_user_id][movie])

            # Calculate similarity

            # Ensure vectors have items in them
            if len(user_vector) > 1:
                numerator = numpy.dot(user_vector, target_user_vector)
                denominator = numpy.sqrt(numpy.dot(user_vector, user_vector)) * numpy.sqrt(numpy.dot(target_user_vector, target_user_vector))

                # Set smiilarity equal to pearson coefficient
                if (denominator != 0):
                    similarities[user] = numerator / denominator

        #Remove self        
        del similarities[target_user_id];

        if (k_value > 0):
            # Choose K closest users
            closest_users = dict(Counter(similarities).most_common(k_value))

        # Calculate weighted average
        total = 0
        weights = 0

        # If a k value was specified, only use closest users for prediction
        if k_value > 0:
            user_set = closest_users
        # If k = 0, use all users for prediction
        else:
            user_set = similarities

        # Iterate over users, summing weights and ratings
        for user in user_set:
            # Only include in prediction if target_movie was rated
            if target_movie_id in norm_data[user]:
                # Total = sum of (rating)(similarity)
                total += similarities[user]*norm_data[user][target_movie_id]

                # Weight = sum of similarities
                weights += similarities[user]

        # If none of the users rated the target film, default to 3.0
        if weights == 0:
            prediction = 0
        # Otherwise, calculate the prediction
        else:
            prediction = total / weights

        prediction = (0.5 * ((prediction + 1) * (5 - 1))) + 1

        print('Command'.ljust(11) + ' = ' + command)
        print('Training'.ljust(11) + ' = ' + sys.argv[2])
        print('Algorithm'.ljust(11) + ' = ' + algorithm)
        print('K'.ljust(11) + ' = ' + str(k_value))
        print('UserID'.ljust(11) + ' = ' + str(target_user_id))
        print('MovieID'.ljust(11) + ' = ' + str(target_movie_id))
        print('Prediction'.ljust(11) + ' = ' + str(prediction))

# Evaluate ratings using the chosen algorithm
if command == 'evaluate':
    print('This command has been temporarily removed')
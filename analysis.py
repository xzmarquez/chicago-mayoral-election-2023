"""
Title: analysis.py
Date: November 2023
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # for regression with R syntax

PATH = '/Users/xzmarquez/Documents/GitHub/Personal/chicago-mayoral-election-2023/final-project/data'

############################## PREP ANALYSIS DATA #############################

def load_analysis_data():
    '''Load demographic and election data & perform analysis-specific cleaning
    
    Returns
    -------
        demo_vote: data frame to be used for analysis

    '''
    demo_vote = pd.read_csv(os.path.join(PATH,'demo_vote_data.csv'))

    # add binary column for if johnson won the ward
    demo_vote['johnson_won'] = np.where(demo_vote['winner'] == 'Johnson', 1, 0)
    
    # add column for percent black
    demo_vote['Proportion Black'] = demo_vote['Black or African American'] / demo_vote['Total Pop (Race)']

    return demo_vote

demo_vote = load_analysis_data()

############################## LINEAR REGRESSION ##############################

# renaming variables of interest to comply with smf format/syntax
demo_vote = demo_vote.rename(columns={'Proportion Non-White':'perc_non_white',
                                      'Propotion Voting Pop 18-29':'perc_under_30',
                                      'Proportion Rent Burdened':'perc_rent_burdened'})

# regression 1
# y = johnson vote share
# x  = % non-white, % 18-29, % rent burdened
def regression(df, y, x1, x2, x3 = ''):
    if x3 != '':
        reg = smf.ols(f'{y} ~ {x1} + {x2} + {x3}', data = df)
    else:
        reg = smf.ols(f'{y} ~ {x1} + {x2}', data = df)
    result = reg.fit()
    return result.summary()

reg1 = regression(demo_vote,'proportion_johnson','perc_non_white','perc_under_30','perc_rent_burdened')
print(reg1)

# robustness check: are perc_non_white and perc_rent_burdened multicollinear?
# run regression without perc_non_white
reg2 = regression(demo_vote,'proportion_johnson','perc_under_30','perc_rent_burdened')
print(reg2)

# run regression without perc_rent_burdened
reg3 = regression(demo_vote,'proportion_johnson','perc_under_30','perc_non_white')
print(reg3)


############################## NEAREST NEIGHBORS ##############################

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# subset relevant columns to be used in the prediction algorithm
analysisData = demo_vote[['ward_id', 'Proportion White', 'Proportion Black', 
                          'Proportion Rent Burdened', 'Proportion Voting Pop 18-29', 
                          'johnson_won']].dropna().set_index('ward_id')

# split analysis data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(analysisData[['Proportion White', 
                                                                  'Proportion Black', 
                                                                  'Proportion Rent Burdened', 
                                                                  'Proportion Voting Pop 18-29']],
                                                    analysisData['johnson_won'], 
                                                    stratify = analysisData['johnson_won'],
                                                    train_size = 0.7, random_state = 20231128)

# examine number of observations in each partition
print('=== All Observations ===')
print(analysisData['johnson_won'].count())
print('=== Training Partition ===')
print(y_train.count())
print('=== Testing Partition ===')
print(y_test.count())

# check if the proportions of y label (johnson_won) are maintained in the partitions
# proportions of johnson vs vallas should be equal in training and testing partitions bc we stratified by johnson_won
print('=== All Observations ===')
print(analysisData['johnson_won'].value_counts(normalize = True))
print('=== Training Partition ===')
print(y_train.value_counts(normalize = True))
print('=== Testing Partition ===')
print(y_test.value_counts(normalize = True))


# create function to calculate the misclassification rate
# we will classify a ward as voting for johnson if the proportion of johnson_won = 1 among its neighbors 
# is greater than or equal to .5
# we will deem a ward as misclassified if its predicted winner does not match its actual winner 
def nbrs_metric (class_prob, y_obs):
    '''Calculate classification metrics

    Parameters
    ----------
        class_prob: a dataframe of predicted probability for target categories
        y_obs: a series of observed target categories

    Returns
    -------
        mce: misclassificatinon error - proportion of observations that are misclassified
    '''
    nbrs_pred = np.where(class_prob[1].to_numpy() > .5, 1, 0)
    mce = np.mean(np.where(nbrs_pred == y_obs, 0, 1))
    return (mce)


def find_k(max_k):
    ''' Find the optimal number of neighbors among range of 2-10

    Parameters
    ----------
    None
    
    Returns
    -------
    k_result_df: a data frame with misclassification errors for each k value

    '''

    neigh_choice = range(2,max_k,1)

    k_result = []
    for k in neigh_choice:
        neigh = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
        nbrs = neigh.fit(X_train, y_train)

        cprob_train = pd.DataFrame(nbrs.predict_proba(X_train), columns = [0,1])
        mce_train = nbrs_metric(cprob_train, y_train)

        cprob_test = pd.DataFrame(nbrs.predict_proba(X_test), columns = [0,1])
        mce_test = nbrs_metric(cprob_test, y_test)

        k_result.append([k, mce_train, mce_test])

    k_result_df = pd.DataFrame(k_result, columns = ['k', 'MCE_Train', 'MCE_Test'])
    
    return k_result_df

k_result_df = find_k(11)

# plot misclassification rate
# from the plot we can see that 4 neighbors yields the lowest misclassification 
# rate in the testing data
def plot_k(df, max_k):
    neigh_choice = range(2,max_k,1)
    plt.subplots(1, figsize = (10,6), dpi = 200)
    plt.plot(df['k'], df['MCE_Train'], marker = 'o', label = 'Training')
    plt.plot(df['k'], df['MCE_Test'], marker = '^', label = 'Testing')
    plt.title('Johnson_Won-Stratified Random Partition (70/30)')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Misclassification Rate')
    plt.xticks(neigh_choice)
    plt.grid(axis = 'both')
    plt.legend()
    return plt.show()

plot_k = plot_k(k_result_df, 11)


# 
def predict_winners():
    '''Use nearest neighbors algorithm to predict winner of each ward

    Returns
    -------
    prediction_df: data frame with predicted probabilities for each candidate and comparison of actual vs predicted

    '''
    # create the final model with k = 4 and using standard euclidean distance metric
    knn = KNeighborsClassifier(n_neighbors = 4, metric = 'euclidean')

    # train the model
    nbrs_knn = knn.fit(X_train, y_train)

    # use model to predict winner for the wards in the testing data
    prediction = pd.DataFrame(nbrs_knn.predict_proba(X_test), 
                              columns = ['prob_vallas','prob_johnson'], 
                              index = X_test.index.copy())
    prediction['predicted_winner'] = np.where(prediction['prob_johnson'] > prediction['prob_vallas'], 1, 0)

    # merge original data with predictions to compare actual vs predicted winner
    prediction_df = prediction.merge(y_test, how = 'left', on = 'ward_id')
    prediction_df.rename(columns = {'johnson_won':'actual_winner'}, inplace = True)
    prediction_df['correct?'] = np.where(prediction_df['predicted_winner'] == prediction_df['actual_winner'], 1, 0)

    return prediction_df

prediction_df = predict_winners()


# find the nearest neighbors of a specific ward
def find_neighbors(ward_number):
    '''Find the neareast neighbors of a ward

    Parameters
    ----------
        ward_number: the number of a specific ward (integer from 1 to 50)

    Returns
    -------
        ward_neighbors: a dataframe of the 4 nearest neighbors of that ward
    '''
    knn = KNeighborsClassifier(n_neighbors = 4, metric = 'euclidean')
    nbrs_knn = knn.fit(X_train, y_train)
    i = ward_number - 1
    focal = analysisData.iloc[[i]][['Proportion White', 'Proportion Black', 
                                    'Proportion Rent Burdened', 'Proportion Voting Pop 18-29']]
    myNeighborDistance, myNeighborPosition = knn.kneighbors(focal, return_distance = True)
    neighbors_supervised = [X_train.iloc[i] for i in myNeighborPosition][0]
    ward_neighbors = analysisData.iloc[neighbors_supervised.index]
    return ward_neighbors
    
find_neighbors(3)


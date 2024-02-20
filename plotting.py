#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: plotting.py
Date: November 2023
"""
import os
import pandas as pd
import geopandas
import matplotlib.pyplot as plt

PATH = '/Users/xzmarquez/Documents/GitHub/Personal/chicago-mayoral-election-2023/final-project/data'

############################ PREPARE THE DF TO MAP ###########################
def ward_geography():
    """
    Loads Ward geometry and merges it with demographic/voting data 
    """
    #load voting/demographic data from data cleaning csv file
    demo_vote = pd.read_csv(os.path.join(PATH, 'demo_vote_data.csv'))
    
    #load the Ward boundaries shape file
    shp = 'Chicago Ward Boundaries/geo_export_d0c98a2f-b174-4a25-9c07-dbfa2d61ff9f.shp'
    gdf = geopandas.read_file(os.path.join(PATH,shp))
    gdf['ward'] = gdf['ward'].astype(int)
    
    #merge the ward boundaries with the demographic/voting data to use to plot
    gdf = gdf.merge(demo_vote,left_on='ward',right_on='ward_id',how='outer')
    gdf = gdf.drop('ward_id',axis=1)
    
    return gdf

chicago_wards = ward_geography()

############################# WARD WINNER MAP #################################

def vote_map (gdf):
    """
    Creates three maps, one showing which candidate recieved majority vote by 
    Ward, another showing the # of votes cast per ward, the final that shows 
    the proportion of voting-age residents that voted

    """
    fig, ax = plt.subplots(ncols=3,figsize=(30,15),subplot_kw={'aspect':'equal'})
    
    #plotting the candidate that won each ward (>50% vote share)
    ax1 = gdf.plot(column='winner',
                   edgecolor='black',
                   legend=True, 
                   legend_kwds={'fontsize':'xx-large'},
                   ax=ax[0])
    ax1.set_title('Candidate w/ Majority Vote (>50%) \n by Ward', 
                  fontsize=25)
    ax1.axis('off')
    
    #plotting the number of votes cast per ward
    ax2 = gdf.plot(column='votes_total',
                   cmap='Greens', 
                   edgecolor='black',
                   legend=True,
                   ax=ax[1])
    ax2.set_title('Number of Votes \n by Ward', 
                  fontsize=25)
    ax2.axis('off')
    
    #plotting the proportion of voting age residents that cast a vote to see
    #which Wards voted at the highest/lowest rate
    gdf['Proportion_Voted'] = gdf['votes_total']/gdf['Voting-Age Pop (Total)']
    ax3 = gdf.plot(column='Proportion_Voted',
                   edgecolor='black',
                   cmap='Purples',
                   legend=True, 
                   ax=ax[2])
    ax3.set_title('Proportion of Voting-Age Residents\nThat Cast a Vote by Ward', 
                  fontsize=25)
    ax3.axis('off')
    
    fig.suptitle('Chicago Mayoral Runoff Election 2023', 
                 fontsize=30)

    return fig

vote_map = vote_map(chicago_wards)
vote_map.savefig('Ward_Voting.png',bbox_inches='tight', dpi=150)
plt.show()

######################## PROPORTION VOTE SHARE MAP ###########################

def prop_vote_map(gdf):
    """
    Creates a map showing the vote share for each candidate by Ward

    """
    fig, ax = plt.subplots(ncols=2,figsize=(30,15))
    
    #Proportion vote share by Ward for Brandon Johnson
    ax1 = gdf.plot(column='proportion_johnson',
                   cmap='Blues',
                   edgecolor='black',
                   legend=True, 
                   ax=ax[0])
    ax1.set_title('Brandon Johnson Vote Share \n by Ward', fontsize=23)
    ax1.axis('off')
    
    #pProportion vote share by Ward for Paul Vallas
    ax2 = gdf.plot(column='proportion_vallas',
                   cmap='Reds', 
                   edgecolor='black',
                   legend=True,
                   ax=ax[1])
    ax2.set_title('Paul Vallas Vote Share \n by Ward', fontsize=23)
    ax2.axis('off')

    return fig

vote_share_map = prop_vote_map(chicago_wards)
vote_share_map.savefig('Vote_Share.png',bbox_inches='tight', dpi=150)
plt.show()

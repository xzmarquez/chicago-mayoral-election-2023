#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: data_wrangling.py
Date: November 2023
"""

import os
import pandas as pd
from census import Census
import requests
from bs4 import BeautifulSoup
import re
import numpy as np

PATH = '/Users/danya/Documents/GitHub/personal github/final-project-danya-ellen-and-xochi/data'


###################### Census Cook County Demographic Data ####################

#Census API: https://pygis.io/docs/d_access_census.html

def load_county_census():
    """
    Load and clean ACS5 2021 data using the Census API for Cook County, IL 
    on the census block group level
    """
    #Load data for selected variables using Census API
    key = Census('54017d1b3c8a4211ca0a6d5e3f985abba42481f0')
    variables = {'B02001_001E':'Total Pop (Race)',
                 'B02001_002E':'White',
                 'B02001_003E':'Black or African American',
                 'B02001_004E':'American Indian or Alaska Native',
                 'B02001_005E':'Asian',
                 'B02001_006E':'Native Hawaiian or other Pacific Islander',
                 'B02001_007E':'Other Race',
                 'B02001_008E':'Two or More Races',
                 'B03003_001E':'Hispanic or Latino (Total)',
                 'B03003_002E':'Not Hispanic or Latino',
                 'B03003_003E':'Hispanic or Latino',
                 'B25070_001E':'Total Pop (Rent)',
                 'B25070_007E':'30 - 34.9% Income on Rent',
                 'B25070_008E':'35 - 39.9% Income on Rent',
                 'B25070_009E':'40 - 49.9% Income on Rent',
                 'B25070_010E':'>50% Income on Rent',
                 'B29001_001E':'Voting-Age Pop (Total)',
                 'B29001_002E':'18 to 29',
                 'B29001_003E':'30 to 44',
                 'B29001_004E':'45 to 64',
                 'B29001_005E':'65+'
                 }
    census = key.acs5.state_county_blockgroup(fields = ('NAME','B02001_001E',
                                                       'B02001_002E',
                                                       'B02001_003E',
                                                       'B02001_004E',
                                                       'B02001_005E',
                                                       'B02001_006E',
                                                       'B02001_007E',
                                                       'B02001_008E',
                                                       'B03003_001E',
                                                       'B03003_002E',
                                                       'B03003_003E',
                                                       'B25070_001E',
                                                       'B25070_007E',
                                                       'B25070_008E',
                                                       'B25070_009E',
                                                       'B25070_010E',
                                                       'B29001_001E',
                                                       'B29001_002E',
                                                       'B29001_003E',
                                                       'B29001_004E',
                                                       'B29001_005E'),
                                             state_fips = '17',
                                             county_fips = '031',
                                             blockgroup = '*',
                                             year = 2021)
    #convert to dataframe and clean 
    df = pd.DataFrame(census)
    df = df.rename(columns=variables)
    df['GEOID'] = df['state']+df['county']+df['tract']+df['block group']
    df['GEOID'] = df['GEOID'].astype(int)
    df = df.drop(['NAME','state','county','tract','block group'], axis=1)
    return df

cook_county = load_county_census()

######################### Chicago Ward Demographic Data #######################

def chicago_blockgroup_ward_mapping():
    """
    Filter to only the census block groups in Chicago and map/aggregate to Ward
    """

    df = pd.read_csv(os.path.join(PATH, 'block_group_ward_mapping.csv'))
    
    df = df.merge(cook_county,left_on='bg_id',right_on='GEOID',how='left')
    
    df = df.groupby('ward_id').agg({'Total Pop (Race)':'sum',
                                               'White':'sum',
                                               'Black or African American':'sum',
                                               'American Indian or Alaska Native':'sum',
                                               'Asian':'sum',
                                               'Native Hawaiian or other Pacific Islander':'sum',
                                               'Other Race':'sum',
                                               'Two or More Races':'sum',
                                               'Hispanic or Latino (Total)':'sum',
                                               'Not Hispanic or Latino':'sum',
                                               'Hispanic or Latino':'sum',
                                               'Total Pop (Rent)':'sum',
                                               '30 - 34.9% Income on Rent':'sum',
                                               '35 - 39.9% Income on Rent':'sum',
                                               '40 - 49.9% Income on Rent':'sum',
                                               '>50% Income on Rent':'sum',
                                               'Voting-Age Pop (Total)':'sum',
                                               '18 to 29':'sum',
                                               '30 to 44':'sum',
                                               '45 to 64':'sum',
                                               '65+':'sum'}).reset_index()
    #calculate % rent-burdened in each Ward (>30% income on rent)
    df['Total Rent Burden'] = (df['30 - 34.9% Income on Rent']+
                               df['35 - 39.9% Income on Rent']+
                               df['40 - 49.9% Income on Rent']+
                               df['>50% Income on Rent'])
    df['Proportion Rent Burdened'] = df['Total Rent Burden']/df['Total Pop (Rent)']
    
    #calculcate the proportion White/Non-White
    df['Proportion White'] = (df['White']/df['Total Pop (Race)'])
    df['Proportion Non-White'] = 1-df['Proportion White']
    
    #calculate % voting-age pop that falls within different age categories
    df['Proportion Voting Pop 18-29']= df['18 to 29']/df['Voting-Age Pop (Total)']
    df['Proportion Voting Pop 30-44']= df['30 to 44']/df['Voting-Age Pop (Total)']
    df['Proportion Voting Pop 45-64']= df['45 to 64']/df['Voting-Age Pop (Total)']
    df['Proportion Voting Pop 65+']= df['65+']/df['Voting-Age Pop (Total)']
    
    df = df.drop(['30 - 34.9% Income on Rent','35 - 39.9% Income on Rent',
                  '40 - 49.9% Income on Rent','>50% Income on Rent'], axis=1)
    return df

chicago = chicago_blockgroup_ward_mapping()

############################# Mayoral Election Data ###########################

def load_election():
    '''Load and clean election data from 2023 Chicago Mayoral Run-Off Election 

    Parameters
    ----------
    None

    Output
    ------
    DataFrame with vote results for all 50 wards
    '''
    
    # load excel file
    df = pd.read_excel(os.path.join(PATH, 'election_data.xlsx'))

    # drop blank rows
    df = df.dropna(how = 'all', axis = 0)

    # drop 'total' rows
    df = df.drop(df[df['Precinct']=='Total'].index)

    # drop rows with column headers
    df = df.drop(df[df['Ward']=='Ward'].index)
    
    # drop percentage and precinct columns
    perc = ['Precinct','%','%.1']
    df = df.drop(perc, axis=1)
    
    # make columns numeric
    df = df.astype(float)

    # aggregate by ward
    df = df.groupby('Ward').sum().reset_index()
    
    # make new percentage columns
    df['proportion_johnson'] = df['BRANDON JOHNSON '] / df['Votes']
    df['proportion_vallas'] = df['PAUL VALLAS '] / df['Votes']

    # rename columns
    df.columns = ['ward','votes_total','votes_johnson','votes_vallas',
                  'proportion_johnson','proportion_vallas']
    
    return df

election = load_election()

##################### Merging Demographic & Election Data #####################

def demo_vote_merge(df1,df2):
    # merge data frames
    combined_df = df1.merge(df2,
                            how='left',
                            left_on='ward_id',
                            right_on='ward')

    # drop extra ward column
    combined_df = combined_df.drop('ward',axis=1)
    
    #Determine who won each ward
    combined_df['winner'] = np.where(combined_df['proportion_johnson']>0.5,'Johnson','Vallas')

    # writing to a csv
    file = os.path.join(PATH,'demo_vote_data.csv')
    combined_df.to_csv(file,index=False)
    return combined_df
    
demo_vote = demo_vote_merge(chicago,election)


###################### Web Scraping for Text Analysis #########################

def get_wikipedia_data(url):
    """
    Retrieves data from the 2023 Chicago Mayoral Race Wikipedia page.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content,'html.parser')
    return soup


def extract_cite_note_elements(soup):
    """
    Extracts cite note elements from a BeautifulSoup object.
    """
    cite_note_elements = soup.find_all(lambda tag: tag.has_attr('id') and tag['id'].startswith('cite_note-'))
    return cite_note_elements


def clean_text(cite_note_elements):
    """
    Cleans up text by extracting and joining elements.
    """
    result = [element.get_text(strip=True) for element in cite_note_elements]
    cleaned_text = '\n'.join(result)
    return cleaned_text


def extract_information(cleaned_text):
    """
    Extracts information from cleaned text using regex patterns.
    """
    title_pattern = re.compile(r'"(.*?)"')
    source_pattern = re.compile(r'"\.(.*?)\.')

    split_text = cleaned_text.split('\n')

    titles = []
    sources = []

    for element in split_text:
        title_match = title_pattern.search(element)
        source_match = source_pattern.search(element)

        if title_match and source_match:
            title = title_match.group(1).strip()
            source = source_match.group(1).strip()

            titles.append(title)
            sources.append(source)

    return titles, sources


def save_to_csv(data, filename):
    """
    Saves data to a CSV file.
    """

    filepath = os.path.join(PATH,filename)
    data.to_csv(filepath,index=False,encoding='utf-8')


def process_data():
    """
    Processes data from the Wikipedia page, extracts information, and saves it to a CSV file.
    """
    # Specify the URL
    url = "https://en.wikipedia.org/wiki/2023_Chicago_mayoral_election"

    # Extract data from Wikipedia
    soup = get_wikipedia_data(url)
    cite_note_elements = extract_cite_note_elements(soup)
    cleaned_text = clean_text(cite_note_elements)

    # Extract information
    titles, sources = extract_information(cleaned_text)

    # Save the extracted information to a CSV file
    references_csv = pd.DataFrame(list(zip(titles,sources)), columns=['Title','Source'])
    save_to_csv(references_csv,'raw_data_sources.csv')


def manual_cleanup_sources(df):
    """
    Manual clean-up of some news sources in filtered_df.
    """
    # Manual clean-up of sources
    df.at[29,'Source'] = 'Block Club Chicago'
    df.at[31,'Source'] = 'WTTW News'
    df.at[87,'Source'] = 'NBC Chicago'
    df.at[101,'Title'] = "Vallas or Johnson? What Black Chicago's political endorsements say about our leaders"
    df.at[103,'Source'] = 'WTTW News'
    df.at[84,'Source'] = 'Chicago NOW PAC'


def replace_sources(df):
    """
    Additional source replacements in filtered_df.
    """
    # Additional source replacements for news outlets with different naming
    # conventions but are the same source 
    df['Source'] = df['Source'].replace('NBC News5 Chicago','NBC Chicago')
    df['Source'] = df['Source'].replace(['abc7chicago','ABC News7'], 'ABC7 Chicago')
    df['Source'] = df['Source'].replace('Politico','POLITICO')
    df['Source'] = df['Source'].replace(['CBS NewsChicago','CBS News'], 'CBS News Chicago')
    df['Source'] = df['Source'].replace('Crain Chicago Business',"Crain's Chicago Business")
    df['Source'] = df['Source'].replace(['Fox32 Chicago','Fox 32 Chicago'],'FOX 32 Chicago')
    df['Source'] = df['Source'].replace('Sun-Times Media Wire','Chicago Sun-Times')
    

def filter_and_clean_data():
    """
    Read CSV file into new dataframe, filter and clean data, and return df.
    """
    references_df = pd.read_csv(os.path.join(PATH,'raw_data_sources.csv'))

    # Filtering df for only titles focusing on Brandon Johnson and Paul Vallas
    filtered_df = references_df[references_df['Title'].str.contains('Vallas|Johnson', case=False, na=False)]
    filtered_df = filtered_df.reset_index(drop=True)

    # Manual clean-up of titles
    manual_cleanup_sources(filtered_df)

    # Additional source replacements
    replace_sources(filtered_df)
    
    # Drop a couple rows that are the same headline 
    filtered_df = filtered_df[~filtered_df['Title'].str.contains("03/")]
    
    # Drop duplicates based on both 'Title' and 'Source'
    filtered_df = filtered_df.drop_duplicates(subset=['Title', 'Source'], keep='first')
    filtered_df = filtered_df.reset_index(drop=True)
    
    return filtered_df


if __name__ == "__main__":
    process_data()
    filtered_df = filter_and_clean_data()
    
    save_to_csv(filtered_df,'clean_news_sources.csv')









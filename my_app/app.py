"""
Title: app.py
Date: November 2023
"""
from shiny import App, render, ui, reactive
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

############################## WARD DATA PREP ###############################
PATH = '/Users/danya/Documents/GitHub/personal github/final-project-danya-ellen-and-xochi/data'

def get_clean_ward_data():
    df = pd.read_csv(os.path.join(PATH, 'demo_vote_data.csv'))
    df['Proportion Hispanic/Latino'] = (df['Hispanic or Latino']/
                                        df['Hispanic or Latino (Total)'])
    df['Proportion Not Hispanic/Latino'] = (df['Not Hispanic or Latino']/
                                            df['Hispanic or Latino (Total)'])
    return df

ward_data = get_clean_ward_data()

############################ VOTE SHARE REG PLOT #############################
def variable_regplot(df,choice):
    """
    Creates two regression plots, showing the vote share for Johnson and Vallas 
    based on selected demographic variable selected by the user 
    """
    fig, ax = plt.subplots(2)
    
    data = df[['ward_id','proportion_johnson','proportion_vallas',choice]]
        
    xmin = min(data[choice])
    xmax = max(data[choice])

    #Johnson Vote Share by Variable
    ax1 = sns.regplot(data=data, 
                      x=choice,
                      y='proportion_johnson',
                      scatter_kws={'s':20},
                      color='g',
                      ax=ax[0])
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylabel('Johnson Vote Share')
    ax1.set_xticks([])
    ax1.set_xlabel(None)

    #Vallas Vote Share by Variable
    ax2 = sns.regplot(data=data, 
                      x=choice,
                      y='proportion_vallas',
                      scatter_kws={'s':20},
                      ax=ax[1])
    ax2.set_xlim(xmin,xmax)
    ax2.set_ylabel('Vallas Vote Share')
    ax2.set_xlabel(f'{choice}')

    #Plot Labeling
    fig.suptitle(f'{choice} and Candidate Vote Share by Ward')
    sns.despine()
    return fig

##################### VARIABLE SCATTER PLOTS #################################

def variable_scatter(df,variable1,variable2):
    """
    Creates a scatter plot to visually investigate relationships between
    demographic variables. Users can select two variables to plot.
    """
    df['Ward Winner'] = np.where(df['proportion_johnson']>0.5,'Johnson','Vallas')
    fig, ax = plt.subplots(1)
    ax = sns.scatterplot(data=df,
                         x=variable1,
                         y=variable2,
                         hue='Ward Winner')
    ax.set_title(f'{variable2} by {variable1}')
    ax.legend(title='Ward Winner',
              bbox_to_anchor=(1.05,1),
              loc='upper left',
              borderaxespad=0)
    
    sns.despine()
    return fig

################################ CREATE APP ##################################

app_ui = ui.page_fluid(
    ui.panel_title("Final Project Data & Programming II: Danya, Xochi, Ellen"),
    ui.row(
        ui.layout_sidebar(
            ui.panel_sidebar(
                ui.h6('Please use the dropdowns to select the type of variable\
                      you would like analyze'),
                ui.input_select(id='group',
                                label='Select a Variable Group',
                                choices=['Race',
                                         'Ethnicity',
                                         'Voting Population Age',
                                         'Rent Burden']),
                ui.input_select(id='option',
                                label='Select an Option',
                                choices=[])
                ),
            ui.panel_main(
                ui.output_plot('reg_plot'))
            
        )),
    ui.row(
        ui.layout_sidebar(
            ui.panel_sidebar(
            ui.h6('Plot the relationship between different demographic variables.\
                  Please use the dropdowns to select two variables\
                  you would like analyze'),
            ui.input_select(id='variable1',
                            label='Select Your X Variable',
                            choices=['Proportion Non-White',
                                     'Proportion Hispanic/Latino',
                                     'Proportion Rent Burdened',
                                     'Proportion Voting Pop 18-29',
                                     'Proportion Voting Pop 30-44',
                                     'Proportion Voting Pop 45-64',
                                     'Proportion Voting Pop 65+']),
            ui.input_select(id='variable2',
                            label='Select Your Y Variable',
                            choices=['Proportion Non-White',
                                     'Proportion Hispanic/Latino',
                                     'Proportion Rent Burdened',
                                     'Proportion Voting Pop 18-29',
                                     'Proportion Voting Pop 30-44',
                                     'Proportion Voting Pop 45-64',
                                     'Proportion Voting Pop 65+'])
            ),
        ui.panel_main(
            ui.output_plot('scatterplot')))
        
            ))
        
        
def server(input, output, session):
    
    @reactive.Effect
    def _():
        categories = {'Race':['Proportion White',
                              'Proportion Non-White'],
                      'Ethnicity':['Proportion Hispanic/Latino',
                                   'Proportion Not Hispanic/Latino'],
                      'Voting Population Age':['Proportion Voting Pop 18-29',
                                               'Proportion Voting Pop 30-44',
                                               'Proportion Voting Pop 45-64',
                                               'Proportion Voting Pop 65+'],
                      'Rent Burden':['Proportion Rent Burdened']}
        selection = input.group()
        variables=categories[selection]
        
        ui.update_select(
            'option',
            choices=variables)
    
    @output
    @render.plot
    def reg_plot():
        choice = input.option()
        ax = variable_regplot(ward_data,choice)
        return ax
    
    @output
    @render.plot
    def scatterplot():
        variable1 = input.variable1()
        variable2 = input.variable2()
        ax = variable_scatter(ward_data,variable1,variable2)
        return ax
        
app = App(app_ui, server)

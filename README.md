### Group Members: 
Danya Sherbini (@dsherbini), Xochi Marquez (@xzmarquez), Ellen Vari (@eivari)

### Research Questions:
In this project, we analyze the results of the April 2023 Chicago Mayoral Run-Off election between Brandon Johnson and Paul Vallas. Our primary research questions include: 
* What was the distribution of voting results across Chicago wards?
* What is the correlation between election results and race, age, and rent burden?
* How did the sentiment of news coverage differ between candidates?
* Can we predict the winning candidate for each ward based on a certain set of neighborhood characteristics? 

### Approach: 
_Data Wrangling:_ 

We began by gathering 4 data sets: 
1. Mayoral Election Results by Ward (Source: Chicago Board of Election Commissioners)
2. Chicago Ward Maps (Source: chicago.gov)
3. Demographic data: voting age, rent burden, and race/ethnicity (Source: U.S. Census API)Voting age
4. Election-related article headlines (Source: Wikipedia)

We then mapped demographic data from census block group level to ward level and merged with the ward-level election results. This dataset (demo_vote.csv) was the source for our plotting and our analysis.

_Plotting:_ 

We plotted four static choropleths, including one showing the candidate with the majority vote share, one showing the number of votes per ward, one showing the proportion of voting-age residents per ward, and two showing the vote share of Brandon Johnson and Paul Vallas. 

We also created a shiny dashboard that features a regression plot and a scatter plot. The regression plot shows the relationship between each candidate’s vote share and our demographic variables of interest (race, age, rent burden) by ward. The scatter plot shows the relationship between demographic variables that are used in our analysis. Users can customize these plots by selecting different variables from dropdown menus.

_Text Processing:_ 

We web scraped a list of article headlines (raw_data_sources.csv) from the 2023 Chicago Mayoral Election Wikipedia page. We filtered rows containing headlines related to Johnson and Vallas and manually cleaned some of the headlines and sources. Duplicate entries were removed, and the final filtered and cleaned data frame was saved as 'clean_news_sources.csv.' We then cleaned this data in the text_processing file by removing punctuation, converting to lowercase, tokenizing, and removing stopwords. We calculated the linguistic features using part-of-speech tagging and also generated sentiment analysis using NLTK’s SentimentIntensityAnalyzer() and subjectivity and polarity scores using spacytextblob. The clean data frame includes the original title, source, clean title, wordnet position, sentiment score, subjectivity score, and polarity score. We also used spaCy to tokenize and extract linguistic features, including lowercase tokens and part-of-speech tags related to each headline, and calculated the mean subjectivity and polarity score for each candidate.

_Analysis:_ 

For our analysis, we ran a linear regression featuring the share of votes for Brandon Johnson as our dependent variable and the proportion of non-white residents, proportion of residents aged 18-29, and proportion of residents who are rent burdened (i.e., spend more than 30% of their income on rent) as our independent variables.

We also use the nearest neighbors algorithm to predict the winner of each ward. Our label is johnson_won, a binary variable coded as 1 if Brandon Johnson won the ward and 0 if Paul Vallas won. Our features are the proportion of white residents, proportion of Black residents, proportion of residents aged 18-29, and proportion of residents who are rent burdened. We split our data into 70% training data and 30% testing data and use the KNeighborsClassifier() function from sci-kit learn to train and fit the prediction algorithm.

### Limitations: 
A major limitation of this project is the fact that all of the election data is reported by the City of Chicago at the ward level. This meant that for our analysis and plotting, we aggregated all of our demographic data to the ward level. However, wards are not regularly used for analysis of trends for other city-level data. Rather, the focus of city-wide trends is generally Chicago Community Areas. While there are 50 wards in Chicago, there are 77 established Community Areas. This means that each ward covers a larger geographic area than each Community Area. As a result, our analysis is less nuanced than it would be if we were able to look at the Community Area-level. 

An additional limitation of our approach is that we focused on a single run-off election. This means that our models are not generalizable beyond this specific election cycle. If we had included election data from several mayoral elections, our model would have taken into account variation in candidates and any demographic shifts that may impact voting behaviors. 

### Results: 
_Plotting:_ 

There were three main takeaways from our plotting exercise. The first takeaway comes from the static plots that illustrate voting trends across wards. In the image titled “Ward Voting,” we can see that wards cast votes at variable rates. That is, the north side of the city cast a higher number of votes on the whole, with the exception of one ward on the south-west side which had a high number of votes. This trend generally holds when we then look at the proportion of voting-age residents that cast a vote in the election, with residents of wards on the north-side casting votes at a higher rate. However, one difference that we do see is that there is a ward in the center of the city that casted votes at a higher rate (>50%) than other wards. From this map we can also see that voting rates are relatively low across wards, with several wards with voting rates <30%. 

The second and third main takeaways come from our interactive plots. In the “Shiny App - Final Project” image, you will see two plots. In the plot titled “Proportion Rent-Burdened by Proportion Non-White” we see that there appears to be a positive linear relationship between the two variables. That is, as the proportion of rent-burdened residents increases in a ward, so does the proportion of non-white residents. This is a visual check of the relationship between the two variables, which comes into play in our analysis portion of the project. 

In the “Shiny App - Final Project” image on the plot titled “Proportion White and Candidate Vote Share by Ward,” we can see that the voting patterns for the two candidates appear to be the inverse of one another. That is, for Vallas there is a positive correlation between the proportion of white residents in a ward and the proportion of the vote share that he received. For Johnson on the other hand, there is a negative correlation between the proportion of white residents in a ward and the proportion of the vote share that he received.  

_Text Processing:_ 

Below are the results from using spaCy to obtain the subjectivity and polarity scores for headlines related to Johnson and Vallas.
* Johnson: 
  * Mean Subjectivity: 0.0733 
    * This suggests that the text associated with Johnson is objective or non-subjective. 
  * Mean Polarity: 0.0505 
    * There is a slightly positive sentiment, but the value is close to 0, meaning that the sentiment is mostly neutral. 
* Vallas: 
  * Mean Subjectivity: 0.1214 
    * On average, the text associated with Vallas is more subjective compared to Johnson. 
  * Mean Polarity: 0.016 
    * This suggests a slightly positive sentiment, but the value is close to 0 again, indicating a relatively neutral sentiment. 
 
In summary, Vallas has a higher mean subjectivity than Johnson, indicating that the text associated with Vallas is more subjective. Both have mean polarity values close to 0, indicating similar neutral sentiments from the media for both candidates. 

When looking at the headlines associated with endorsement-related language, Johnson was associated with national politicians and activists, while Vallas had local endorsements. Johnson was described as a "progressive" candidate, but Vallas had more critical language associated with his campaign. 

_Analysis:_ 

For our linear regression, our variables of interest are proportion of non-white residents, proportion of residents aged 18-29, and proportion of residents who are rent burdened. However, our plotting showed us that the proportion of non-white residents and proportion of residents who are rent burdened are highly correlated with each other. In order to test for multicollinearity, we conduct a robustness check to determine whether or not both of these variables should be included in our model. 

The robustness check shows that including only the proportion of non-white residents and the proportion of residents aged 18-29 in the regression yields nearly the same results as the original model. This implies that we do not need to include the proportion of residents who are rent burdened in the model. Due to the multicollinearity between proportion non-white and proportion rent burdened, we choose the model without proportion rent burdened as it yields a lower standard error and is thus more precise.

Our final results suggest that for every percentage point increase in the share of non-white population, the share of votes for Brandon Johnson is expected to increase by 5.8 percentage points. This is statistically significant with a p-value of 0. Additionally, they also suggest that for every percentage point increase in the share of voting age 18-29 population, the share of votes for Brandon Johnson is expected to increase by 7.2 percentage points. This is statistically significant with a p-value of .050. 

For our nearest neighbors prediction algorithm, we first determine the optimal number of neighbors (k). We define the optimal value of k as the one that yields the lowest misclassification rate among the testing data.We deem a ward misclassified if its predicted winner does not match its actual winner. We consider all values of k from 2-10 and find the optimal value to be 4. 

We then train the nearest neighbors model with k = 4 and use the standard euclidean distance as our distance metric. We use the model to output the predicted probability that Johnson will win a ward and the predicted probability that Vallas will win a ward. The ward winner is the candidate who has a higher predicted probability. We then compare the predicted winner to the actual winner from the original data and find that the model correctly predicts 13 out of 15 wards in the testing data – an 86% accuracy rate.


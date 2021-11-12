# **Social context influence on views on climate change**
# Abstract
<p align= "justify"> Nowadays, man-made climate change and global warming have become an emergency that goes far beyond national boundaries and certainly represent one of the greatest challenges humankind has ever faced. In fact, because of direct consequences like heat waves, more frequent and intense storms, rising sea levels, warming of the oceans and many others, humans, but most importantly animals, face new challenges for survival. In order to limit as much as possible the damages, concrete actions need to be undertaken. However, identifying the factors that influence public views on climate change is a crucial step to urge politics to propose solutions on the matter and convince the population to act daily. Furthermore, visibility of the topic in the media is crucial to sensitize the population on the severity of the situation. We therefore decided to first evaluate the occurence of climate change and global warming in the discussed matters on the newspapers. We then analysed the influence of the social context (age, academic degree, profession, political party…) on the opinions of climate change using the QuoteBank dataset to answer several questions listed below.</p>

# Research questions
1.	How much is climate change mentioned in the news ?
2.	Does the occurrence of the matter in quotes increase among the years? Does it correlate with the aggravation of climate change ?
3.	Who talks the most about climate change ?
4.	Does the political party influence the views on climate change ? And what about the academic degree ? And the age ?
5.	Do scientists and politicians agree ?

Some of the papers and articles that we have read to define these questions and get inspiration are listed below:<br />
    - [Climate change, cultural cognition, and media effects: Worldviews drive news selectivity, biased processing, and polarized attitudes](https://doi.org/10.1177%2F0963662518801170)<br />
    - [Analysis: The UK politicians who talk the most about climate change](https://www.carbonbrief.org/analysis-the-uk-politicians-who-talk-the-most-about-climate-change)<br />
    - [The Politics of Climate](https://www.pewresearch.org/internet/wp-content/uploads/sites/9/2016/10/PS_2016.10.04_Politics-of-Climate_FINAL.pdf)<br />
    - [Democratic and Republican Views of Climate Change (2018)](https://climatecommunication.yale.edu/visualizations-data/partisan-maps-2018/)<br />

# Proposed additional datasets
The informations about the frequency of research of a certain topic on google will be retrieved using **Google Trends**. It will give informations about the influence of the social media on the population.

# Methods
**Sorting out quotes based on topic:**<br />
<p align= "justify"> In order to sort the quotes to get only those speaking about climate change, the best method is to search quotes by keywords (climate change, global warming, etc..). Topic classification using machine learning methods is not useful because we will surely get quotes that are completely irrelevant for climate change.</p>

**Sentiment analysis:**<br />
<p align= "justify"> Once the dataset is sorted out, we are interested to see whether the quotes are positive or negative about climate change. This method will lead hopefuly to a better interpretation of the perception of climate change by the speakers. There are multiple pre-trained sentiment analysis model that are available on https://huggingface.co/ and we will need to test them to achieve the best possible accuracy. We think that sentiment analysis on such negative topic like climate change will probably fail to determine the opinion of the speaker about this kind of topic but we will still give it a try.</p>

**Quotebank and wikidata:**<br />
<p align= "justify"> The personal informations about the speakers are essential to perform statistical anaylsis. Thus, the important informations such as the age, the nationality, the gender, etc.. are retrieved from the wikidata and added to the quotebank dataframe corresponding to the qid of the speaker.</p>


# Proposed timeline and internal milestones
|**Week**|**Task**|**Members involved**|
|----|----|----------------|
|**1st-7th November**|1.Definition of project idea|All the team|
|   |2.Data exploration and preprocess of the data|Jeremy|
| |3.Data cleaning and further data handling pipelines|Jeremy, Laurent, Gloria|
|**8th-14th November**|1.Identification of key words for climate change|Marwan, Gloria|
| |2.Extract quotes for QuoteBank related to climate change ([Question 1.](#research-questions))|Jeremy, Laurent|
| |3.Initial analysis ([Question 2.](#research-questions))|Jeremy, Laurent, Marwan|
| |4.Writing of the README|Gloria|
|**29th November-5th December**|1.Sort the identified climate quotes based on the profession of the speaker and evaluate what are the top professions talking about climate change ([Question 3.](#research-questions))|All the team|
|**6th-12th December**|1.Begin the sentiment analysis|Laurent|
| |2.Creation of the website for the datastory|Marwan, Gloria|
| |3.Finish sentiment analysis and evaluate the influence of political party, age and academic degree of speaker ([Question 4.](#research-questions))|Jeremy, Laurent|
| |4.Do politicians and scientists agree ? ([Question 5.](#research-questions))|Jeremy, Laurent|
|**13th-17th December**|1.Conclude and resume findings|All the team|
| |2.Finish datastory|Gloria, Marwan|
| |3.Update README|Gloria|


# Organization within the team
**Gloria**<br />
    Project description, writing of the README file, research questions definition, web datastory and data visualisation.<br />
**Jeremy**<br /> 
    Preprocess of the data, data cleaning, data handling, data exploration and general coding.<br />
**Laurent**<br />
    Performing of the initial analysis, sentiment analysis, implementation of the diverse methods, helps Jeremy on the data handling task.<br />
**Marwan**<br />
    Identification of keywords on climate change, data visualisation with Gloria, realisation of the web datastory.<br />


# Questions for the TAs

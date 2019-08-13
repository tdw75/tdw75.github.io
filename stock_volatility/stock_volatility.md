## This can be your internal website page / project page

**Project description:** This was a group research project I did as part of a financial econometrics course at the University of Sydney. As it was a group project, I obviously worked with colleagues on this. The majority of the theoretical background research, stock data collection through the Bloomberg terminals, and financial analysis was done by my group mates. I did all of the coding, including the sentiment analysis and time series modelling. As such, I'll be focusing on that here. The inference was done as a group but I'll go through that as well.

### 1. Introduction and Data Description

The goal of this project is to try to analyse the relationship between sentiment towards a company and the volatility of its stock price. More specifically, we wanted to see the effect that sentiment during the overnight period when the stock market is closed on the volatility at market open the next day. The company we used as an example is Tesla, which as a famously volatile stock should allow for some interesting analysis.

To do this, we clearly need a few different pieces of information. First, we need data on TSLA stock as well as a benchmark to compare it against. Second, we need some covariates to avoid problems such as omitted variable bias, whereby information of an unseen variable is present in the variable of interest, which makes the variable of interest appear to have more explanatory power than it does in reality. Finally, we of course need some measure of sentiment towards Tesla.

For the financial data we used Bloomberg terminals as this allowed us to get tick-by-tick data, i.e. a record of every transaction made. From here we got TSLA stock data for the first 30 mins of each day between 31 October 2018 and 30 April 2019. Unfortunately, the terminals didn't allow us to download data from earlier than 31 October, which limited the time frame of the analysis. As a benchmark we used the S&P500, which is the standard market index for the US stock market. We got S&P500 data for the same time frame.



### 2. Twitter Scraping

To start off with I used the Twitter API to scrape tweets. But I quickly found out that with the API you can only download tweets from the last week, which is a bit useless for this analysis. Luckily, I stumbled across a brilliant package called *GetOldTweets3* which makes it very easy to scrape tweets from any time frame. Using this, I downloaded a year's worth of tweets with the hashtag *#tesla*. The code for this is below

```python
import pandas as pd
import numpy as np
import GetOldTweets3 as got3

tweet_criteria = got3.manager.TweetCriteria().setQuerySearch('#tesla').setSince("2018-05-01").setUntil("2018-05-02")
tweets_raw = pd.DataFrame(got3.manager.TweetManager.getTweets(tweet_criteria))

# extract the text and date of the tweet into a format that we can read/analyse
tweets = pd.DataFrame(columns = ['text', 'date'])
tweets['text'] = tweets_raw[0].apply(lambda x: x.text)
tweets['date'] = tweets_raw[0].apply(lambda x: x.date)

# change tweet timezone from GMT to New York time
tweets = tweets.set_index('date')
tweets = tweets.tz_convert('US/Eastern')
```



### 3. Data Cleaning


```python
print(tweets.isna().sum())
tweets.describe()
```

```python
tweets = tweets.drop_duplicates()
```

Write a function to remove non-punctuation characters

```python
def strip_charachters(string):
    for char in '@#':  
        string = string.replace(char,'')
    return string
tweets['stripped_text'] = tweets['text'].apply(strip_charachters)
```

```python
tweets['language'] = tweets['stripped_text'].apply(langid.classify)
tweets['language'] = tweets['language'].apply(lambda x: x[0])
tweets = tweets[tweets['language'] == 'en']
tweets.head()
```

```python
tweets = tweets.drop(['text', 'language'], axis=1)
tweets.columns = ['text']
```

### 4. Sentiment Analysis and Further Data Preparation

```python
analyser = SentimentIntensityAnalyzer()
scores = pd.DataFrame(tweets['text'].apply(analyser.polarity_scores))
for word in ['neg', 'neu', 'pos', 'compound']: 
    tweets[word] = [d[word] for idx, d in scores.text.items()]
daily_sentiment = tweets['compound'].groupby(pd.Grouper(freq='D')).mean()
```

### 5. Model Estimation


### 5. Inference and Analysis

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

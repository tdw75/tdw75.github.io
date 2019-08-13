## This can be your internal website page / project page

**Project description:** This was a group research project I did as part of a financial econometrics course at the University of Sydney. As it was a group project, I obviously worked with colleagues on this. The majority of the theoretical background research, stock data collection through the Bloomberg terminals, and financial analysis was done by my group mates. I did all of the coding, including the sentiment analysis and time series modelling. As such, I'll be focusing on that here. The inference was done as a group but I'll go through that as well.

### 1. Introduction and Data Description

The goal of this project was to try to analyse the relationship between sentiment towards a company and the volatility of its stock price. More specifically, we wanted to see the effect that sentiment during the overnight period when the stock market is closed on the volatility at market open the next day. The example company here is Tesla, which is listed on the NASDAQ as *TSLA*. Tesla is a famously volatile stock so it should allow for some interesting analysis. This process, however, can be applied to any company.

To do this, we clearly need a few different pieces of information. First, we need a measure of volatility for TSLA stock as well as a benchmark to compare it against. Second, we need some covariates to avoid problems such as omitted variable bias, whereby information of an unseen variable is present in the variable of interest, which makes the variable of interest appear to have more explanatory power than it does in reality. Finally, we of course need some measure of sentiment towards Tesla.

For the measure of volatility, we decided on a metric called realised volatility. As volatility only shows variation at a single point in time, realised volatility allows us to measure the accumulated volatilty in a particular time frame. The calculation for this is shown later on. 

To calculate this, we needed financial data. We got this from Bloomberg terminals as this allowed us to get tick-by-tick data, i.e. a record of every transaction made. From here we got TSLA stock data for the first 30 mins of each day between 31 October 2018 and 30 April 2019. Unfortunately, the terminals didn't allow us to download data from earlier than 31 October, which limited the time frame of the analysis. The standard market index for the US stock market is the S&P500, so we used that as our benchmark and downloaded got S&P500 data for the same time frame.

For the second component we selected two covariates, *Press Release* and *Google Trends*. *Press Release* is a dummy variable of whether Tesla released a press report on that day, we chose this as company press releases, particularly after big events, are likely to influence public sentiment. *Google Trends* is a measure of Tesla's search engine popularity on each day, providing a good proxy for publc engagement with the company.

Finally, we need a measure of public sentiment towards Telsa. To calculate this, we analysed tweets that hashtagged Tesla and used a natural language processing technique called a sentiment analysis to determine if positive or negative language was used when referencing Tesla. The steps for this are shown below.

### 2. Twitter Scraping

To start off with I tried to use the Twitter API to scrape tweets. But I quickly found out that with the API you can only download tweets from the last week, which is a bit useless for this analysis. Luckily, I stumbled across a brilliant package called *GetOldTweets3*, which makes it very easy to scrape tweets from any time frame. Using this, I downloaded a year's worth of tweets with the hashtag *#tesla*. The code for this is below

```python
import pandas as pd
import numpy as np
import GetOldTweets3 as got3

tweet_criteria = got3.manager.TweetCriteria().setQuerySearch('#tesla').setSince("2018-05-01").setUntil("2018-05-02")
tweets_raw = pd.DataFrame(got3.manager.TweetManager.getTweets(tweet_criteria))

# extract the text and date of the tweet
tweets = pd.DataFrame(columns = ['text', 'date'])
tweets['text'] = tweets_raw[0].apply(lambda x: x.text)
tweets['date'] = tweets_raw[0].apply(lambda x: x.date)

# change tweet timezone from GMT to New York time where the NASDAQ is located
tweets = tweets.set_index('date')
tweets = tweets.tz_convert('US/Eastern')
```

### 3. Data Cleaning

With the tweets scraped

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
import langid

tweets['language'] = tweets['stripped_text'].apply(langid.classify)
tweets['language'] = tweets['language'].apply(lambda x: x[0])
tweets = tweets[tweets['language'] == 'en']
tweets.head()
```

```python
tweets = tweets.drop(['text', 'language'], axis=1)
tweets.columns = ['text']
```

### 4. Sentiment Analysis and Realised Volatility

```python
import nltk.sentiment.sentiment_analyzer import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
scores = pd.DataFrame(tweets['text'].apply(analyser.polarity_scores))
for word in ['neg', 'neu', 'pos', 'compound']: 
    tweets[word] = [d[word] for idx, d in scores.text.items()]
daily_sentiment = tweets['compound'].groupby(pd.Grouper(freq='D')).mean()
```

```python
analyser = SentimentIntensityAnalyzer()
scores = pd.DataFrame(tweets['text'].apply(analyser.polarity_scores))
for word in ['neg', 'neu', 'pos', 'compound']: 
    tweets[word] = [d[word] for idx, d in scores.text.items()]
daily_sentiment = tweets['compound'].groupby(pd.Grouper(freq='D')).mean()
```

```python
import matplotlib.pyplot as plt

daily_sentiment1 = daily_sentiment*100
daily_sentiment1.plot(figsize=(16,5))
plt.title('Daily Twitter sentiment towards Tesla')
plt.hlines(0, xmin=min(daily_sentiment1.index), xmax=max(daily_sentiment1.index), linestyles='dashed')
plt.xlabel('Date')
plt.ylabel('Sentiment')

plt.show()
```

<img src="images/Daily Twitter sentiment.png?raw=true"/>

```python
def overnight_sentiment(data):
    time_1 = data.index.indexer_between_time('00:00:00', '9:30:00')
    morning = data.iloc[time_1]
    time_2 = data.index.indexer_between_time('16:30:00', '23:59:59')
    evening = data.iloc[time_2]
    
    sentiment = pd.DataFrame(morning['compound'].groupby(pd.Grouper(freq='D')).mean())
    sentiment.columns = ['morning']
    sentiment['evening'] = evening['compound'].groupby(pd.Grouper(freq='D')).mean()
    sentiment['day'] = (sentiment['morning'] + sentiment['evening'].shift())/2
    
    return sentiment
    
def realised_volatility(data, end='10:00:00'):
    
    data.index = data['Dates']
    time = data.index.indexer_between_time('9:30:00', end)
    data = data.iloc[time]
    data = data.reset_index(drop=True)
    
    data['Previous_price'] = data['Price'].shift()
    data['Sq_log_return'] = np.log1p(data['Price']/data['Previous_price'])**2
    
    data.index = data['Dates']
    data = data.dropna().groupby(pd.Grouper(freq='D')).sum()
    data = data[data.index.dayofweek < 5] # select weekdays only
    data = np.sqrt(data['Sq_log_return'])
    data = data[data!=0]
    
    return data
```

```python
excel = pd.ExcelFile('Collated Opening 60.xlsx')
tsla_oct = pd.read_excel(excel, 'TSLA')
tsla_april = pd.read_excel(excel, 'TSLA Apr')

S_P_index = pd.read_excel(excel, 'S&P500')
S_P_index = S_P_index[['Dates', 'Price']]

tsla_price = pd.concat([tsla_oct, tsla_april], axis=0)
tsla_price = tsla_price[['Dates', 'Price', 'Size']]

tsla = realised_volatility(tsla_price)
S_P = realised_volatility(S_P_index)
overnight = overnight_sentiment(tweets)
overnight.index = overnight.index.strftime('%Y-%m-%d')
covariates = pd.read_excel('Covariates data.xlsx', index_col='Day')

reg_data = pd.merge(pd.DataFrame(tsla), pd.DataFrame(S_P), how='inner', left_index=True, right_index=True)
reg_data = pd.merge(reg_data, pd.DataFrame(overnight['day']), how='inner', left_index=True, right_index=True)
reg_data.columns = ['TSLA', 'S&P500', 'Sentiment']
reg_data = pd.merge(reg_data, covariates, how='inner', left_index=True, right_index=True)
reg_data['Sentiment'] = reg_data['Sentiment']*100
```

### 5. Model Estimation


```python
months = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

fig, ax = plt.subplots(1,1, figsize = [16,5])

ax.plot(y)
ax.set_title("Difference between the realised volatility of TSLA and S&P500 in the first 30 minutes of each day")
ax.set_ylabel("Realised volatility difference")
ax.set_xlabel("Date")
ax.set_xticks(ax.get_xticks()[::21])
ax.set_xticklabels(months)

plt.show()
```

<img src="images/Realised volatility.png?raw=true"/>

```python
import statsmodels.api as sm

y = reg_data['TSLA'] - reg_data['S&P500']
X = np.arange(0, len(y))
X = sm.add_constant(X)
model = sm.OLS(y, X)
res = model.fit()
detrended_y = y - res.

fig, ax = plt.subplots(1,1, figsize = [16,5])

ax.plot(detrended_y)
ax.set_title("Difference between the realised volatility of TSLA and S&P500 in the first 30 minutes of each day")
ax.set_ylabel("Realised volatility difference")
ax.set_xlabel("Date")
ax.set_xticks(ax.get_xticks()[::21])
ax.set_xticklabels(months)

plt.show()
```

<img src="images/Detrended realised volatility.png?raw=true"/>

**Ljung-Box test*

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

LB_test = acorr_ljungbox(y, lags=20)[1]

fig, ax = plt.subplots(1,1, figsize = [16,5])

ax.scatter(np.arange(1,21), LB_test, label = 'p-value')

ax.set_title("Ljung-Box Test Results")
ax.set_ylabel("Lag Order")
ax.set_ylabel("p-value")
ax.set_ylim([0,0.1])

ax.hlines(0.05, 0, 20, linestyles='dashed', label = 'alpha')
plt.legend()

plt.show()
```

<img src="images/Realised volatility Ljung-Box test.png?raw=true"/>

```python
from statsmodels.tsa.stattools import adfuller

unit_root_test = adfuller(detrended_y)
unit_root_test[1]
```



```python
from statsmodels.tsa.arima_model import ARIMA

P = 3
Q = 2

aic = pd.DataFrame(0, index=['AR('+str(i).zfill(1)+')' for i in range(0,P)], 
                   columns=['MA('+str(i).zfill(1)+')' for i in range(0,Q)])

for p in range(0,P):
    for q in range(0,Q):
        y = detrended_y
        X = reg_data[['Sentiment', 'Press Release','Google trends']]
        model = ARIMA(endog=y, exog=X, order=(p,0,q))
        arma = model.fit(method='mle')
        aic.iloc[p,q] = round(arma.aic,3)
        
aic
```

```python
y = detrended_y 
X = reg_data[['Sentiment', 'Press Release','Google trends']]
model = ARIMA(endog=y, exog=X, order=(1,0,0))
armax = model.fit(method='mle')
print(arma`x.summary())
```

### 6. Inference and Analysis

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

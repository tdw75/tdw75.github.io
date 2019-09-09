### Is the Australian Broadcasting Corporation biased against conservatives?

Many non-Australians may not know this, but Australia's media (particularly print media) is generally quite conservative, newspapers owned by Rupert Murdoch account for almost 60% of Australian daily newspaper circulation. The publicly owned ABC is an exception to this and, as a result, is often accused by conservative politicians of having a left-wing bias. Numerous inquiries and investigations have been made into this claim over the last few years, all of which concluded that the ABC is in fact not biased. However, the accusations from right-leaning Australians persist. So is this true? Is the ABC really biased against conservatives? As the ABC is a tax funded intitution, this is an important and interesting question, one which I wanted to try to answer.

For those that are unfamilar with Australian politics, we have two major parties - the centre-left Australian Labor Party (note the different spelling of 'labour') and the conservative Coalition, comprised of the (ironically named) Liberal Party and the National Party. The dataset I used was one containing the headlines of every ABC news article from 19 February 2003 until 31 December 2017. My thinking was that by comparing news articles about the two major parties I could gain an insight into how the ABC reports on each party. The dataset contains 1,103,663 different headlines, an average of roughly 190 per day. The series below shows the volume of articles throughout the time period with markers at each election and change of prime minister. 

*******

We can see...

New let's move onto analysing the headlines concerning each party. The plan is to extract all of the articles about Labor and Coalition and compare how positive or negative the coverage of each party is to see if the parties are treated differently. Particularly in the era of clickbait journalism, a headline may not always be completely align with the content of article. However, as the headline is the first (and sometimes only) thing that people read, it plays a major role in shaping public opinion about the subject of the article. So while an analysis of the complete text of the articles may give a more comprehensive view of any bias, the headlines alone will still give us a good insight into XXXXX, especially considering the large amount of data.

To accomplish this task I will be running a sentiment analysis on each of the headlines. There are two algorithms I am using here. The first is the VADER sentiment analysis in the NLTK module in Python. VADER was originally designed for short social media posts for which its performance was shown to be very good during academic research. Given that news headlines are similarly short, it should also perform quite well here. The second is the sentiment analysis tool in the textblob Python module. This takes a different approach using a Naïve Bayes classifier that has been trained on a large dataset of XXXXXXX.

To extract the Labor and Coalition news headlines I selected all of the articles that contained the following key words:  

For Labor:  
- References to the party (Labor, ALP)
- Surnames of party leaders (Crean, Latham, Beazley, Rudd, Gillard, Shorten)
- Surnames of selected important ministers and treasurers (Swan, Bowen)  
For the Coalition:   
- References to the party (Coalition, LNP, Liberals, National Party, Liberal Party, Libs)
- Surnames of party leaders (Howard, Nelson, Turnbull, Abbott)
- Surnames of selected important ministers and treasurers (Costello, Morrison, Joyce)

The selection resulted in 17,389 articles about Labor and 15,626 about the Coalition. And while this is most likely not an exhaustive list of all references to each party, it will definitely cover the vast majority. 

### VADER Sentiment Analysis

After running the VADER sentiment analysis on each headline within the two groups, I received a sentiment score between -1 and 1 for each. Rather than just leaving this at a numerical score, I categorised them into very negative (score<-0.5), negative (-0.5>score>0), neutral (score=0), positive (0>score>0.5), and very positive (score>0.5). This gave the following distribution:



### TextBlob Sentiment Analysis

The next analysis is with TextBlob. Each headline was again given a sentiment score between -1 and 1 and I categorised them in the saem way as above, which resulted in the following distribution:


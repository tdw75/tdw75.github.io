### Is the Australian Broadcasting Corporation biased against conservatives?

Many non-Australians may not know this, but Australia's media (particularly print media) is generally quite conservative, newspapers owned by Rupert Murdoch account for almost 60% of Australian daily newspaper circulation. The publicly owned ABC is an exception to this and, as a result, is often accused by conservative politicians of having a left-wing bias. Numerous inquiries and investigations have been made into this claim over the last few years, all of which concluding that the ABC is in fact not biased. However, the accusations from right-leaning Australians persist. So is this true? Is the ABC really biased against conservatives? This is an interesting and interesting question, one which I wanted to try to answer.

For those that are unfamilar with Australian politics, we have two major parties - the centre-left Australian Labor Party and the conservative Coalition, comprised of the (ironically named) Liberal Party and the National Party. The dataset I used was one containing the headlines of every ABC news article from **19 February 2003 until 2 January 2018**. My thinking was that by comparing news articles about the two major parties I could gain an insight into how the ABC reports on eaceh party. 

The dataset contains **XXX,XXX** different headlines, an average of roughly **XXX** per day. The series below shows the volume of articles throughout the time period with markers at each election and change of prime minister. 

*******

We can see...

New let's move onto analysing the headliens concerning each party. The plan is to extract all of the articles about Labor and Coalition and compare how positive or negative the coverage of each party is to see if the parties are treated differently. Obviously, a headline may not always be completely revealing of the tone of article, but in the absence of the complete text of each article, they should still give us a reasonable insight into XXXXXXX, especially given the large amount of data available. To accomplish this task I will be running a sentiment analysis on each of the headlines. There are two algorithms I am using here. The first is the VADER sentiment analysis in the NLTK module in Python. VADER was originally designed for short social media posts for which its performance was shown to be very good during academic research. Given that news headlines are similarly short, it should also perform quite well here. The second is the sentiment analysis tool in the textblob Python module. This takes a different approach using a Naïve Bayes classifier that has been trained on a large dataset of XXXXXXX.

To extract the Labor and Coalition news headlines I did the following

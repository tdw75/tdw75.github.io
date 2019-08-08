## A statistical machine learning based pricing recommendation system for Airbnb listings in Melbourne

**Project description:** This project was completed as an assessment piece for a machine learning and data mining unit at the University of Sydney. For this  assignment, we were given a data set of AirBnB listings in Melbourne and were tasked with using statistical learning methods to build a pricing recommendation system. The project gave me a good opportunity to work with a data set that required a fair bit of wrangling and feature engingeering and then to implement a variety of different machine learning algorithms.

A description of the training set can be found in the appendix.

### 1. Exploratory Data Analysis

The training set contains 7,000 listings and 28 independent variables as well as the target variable (price). The listing prices ranged from $13 per night to $5,000, with a mean of $152.19 and a median of $119.00. This implies a relatively significant right skew, which can be seen in distribution below.

<img src="images/Response_distribution.png?raw=true"/>

The heavy right skew and presence of outliers are typical of price data such as this and can cause some problems during estimation. As such, a log-transformation of the data was performed, which resulted in a much more symmetrical and "modelling-friendly" distribution. 

<img src="images/Log_response_distribution.png?raw=true"/>

Of the independent variables, six were categorical, and would thus require further feature engineering in order to be useable. Additonally, latitude and longitude offer no information in their current state; further processing would be required to derive useful insights. A pairwise comparison of each (numerical) independent variable with respect to the price/log price revealed that the majority of variables do not share a linear relationship with the target variable.

<img src="images/______.png?raw=true"/>

With a number of very similar independent variables in the dataset, high multicollinearity was likely to pose a problem. The correlation between each feature was examined through a correlation matrix to better understand the underlying relationships that exist within the dataset.

<img src="images/Correlation_matrix.png?raw=true"/>

As seen above, the review scores display a high positive correlation. This is logical, one would expect that a listing that is reviewed highly in one area would often also score highly in others. Additionally, review_score_rating is an aggregation of the rest of the scores; this implies perfect multicollinearity, which would need to be addressed through feature engineering. 

Variables to do with the size of the property (accomodates, bedrooms, bathrooms, beds, security_deposit, and cleaning_fee) are also highly correlated.

Based off the insights gained from the EDA, it was hypothesised that due to the non-linearity and relatively high dimensionality of the data, linear models could struggle to capture the various relationships between the variables and would therefore perform poorly during prediction. Instead, it was thought that non-linear methods would be superior in this case. The hypothesis is challenged and tested during the model estimation process.

### 2. Data Cleaning, Processing, and Wrangling

As seen in the dataset description, there were a number of variables with missing values. To remove the problem of missing data, observations with missing values could obviously be merely removed from the dataset. However, unless the observation is so sparsely populated that it doesn't offer much information, it's generally not ideal to delete data. Instead, the missung values could be imputed with another value.

There are a number of ways to approach data imputation. As a basic level, you can just replace the missing values with some measure of central tendency (mean, median, etc.). For time series data, something like linear interpolation is popular. Then there is also the possibilty to create a model to predict the missing values, and then use these predicted values in the estimation of the target variable.

In our case, there were three variables (bedrooms, bathrooms, and beds) that had a trival number of missing values (<7). Here, we just imputed the missing values with the respective median. That left ten variables with missing values - security_deposit, cleaning_fee, reviews_per_month and each of the seven review metrics - each with more than 1,000. Given the large number of missing values, filling them all with the same value is less likely to produce good predictions. Therefore, we decided to predict these values. After a bit of research, we found that KNN is a popular method for data imputation due to ease of implementation and relatively good results. The underlying rationale for KNN imputation is that a point value can be approximated by using the values of the points that are closest to it, based on the other features within the data set (excluding the features that are missing large portions of data themselves). This method only requires the specification of the number of nearest neighbours (a ‘k’ value), and a distance metric. In the first instance, 5-fold cross-validation was used (taking computational cost into consideration) to determine an optimal k value of 17. In determining the distance matrix, it is important to first observe the type of data that is missing. All the features missing values were numerical and so the Euclidean distance metric was used. Another important consideration was that due to the sheer volume of missing data entries, and the high dimensionality of the data, many missing values would not have enough neighbours to deliver reliable predictions. To account for this, if more than 50% of the nearest neighbours had missing values in the column of interest, the observation was left unfilled. The first run of KNN imputation filled the vast majority of missing values (e.g. 1,000 in the case of security_deposit). With many more values now populated, the process was repeated. However, after two iterations the algorithm was stopped due to diminishing effectiveness, computational cost, and so as to not compound any estimation bias. As there were only a relatively small number of missing data points remaining, they were filled using the median.

### 3. Feature Engineering  

### 4. Modelling

### 5. Analysis

### 6. Appendix

### 7. Limitations, Comments, and Future Work 

Feature |	Non-null Values |	Data Type
--------|-----------------|----------
Id	| 7000	| Integer (64)
Host is super host	| 7000 |	Object
Host total listings count	| 7000	| Integer (64)
Host identity verified	| 7000	| Object
Latitude	| 7000	| Float (64)
Longitude	| 7000	| Float (64)
Property Type	| 7000	| Object
Room type	| 7000	| Object
Accommodates	| 7000	| Integer (64)
Bathrooms	| 6993	| Float (64)
Bedrooms	| 6999	| Float (64)
Beds	| 6997	| Float (64)
Security deposit	| 5082	| Float (64)
Cleaning fee	| 5608	| Float (64)
Extra people	| 7000	| Integer (64)
Minimum nights	| 7000	| Integer (64)
Maximum nights	| 7000	| Integer (64)
Number of reviews	| 7000	| Integer (64)
Review score rating	| 5646	| Float (64)
Review score accuracy	| 5638	| Float (64)
Review score cleanliness	| 5642	| Float (64)
Review score check-in	| 5635	| Float (64)
Review score communication	| 5640	| Float (64)
Review score location	| 5636	| Float (64)
Review score value	| 5634	| Float (64)
Instant bookable	| 7000	| Object
Cancellation policy	| 7000	| Object
Reviews per month	| 5769	| Float (64)

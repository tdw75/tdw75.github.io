## A statistical machine learning based pricing recommendation system for Airbnb listings in Melbourne

**Project description:** This project was completed as an assessment piece for a machine learning and data mining unit at the University of Sydney. For this  assignment, we were given a data set of AirBnB listings in Melbourne and were tasked with using statistical learning methods to build a pricing recommendation system. 

A description of the training set can be found in the appendix.

### 1. Exploratory Data Analysis

The training set contains 7,000 listings and 28 independent variables as well as the target variable (price). The listing prices ranged from $13 per night to $5,000, with a mean of $152.19 and a median of $119.00. This implies a relatively significant right skew, which can be seen in distribution below.

<img src="images/Response_distribtion.png?raw=true"/>

The heavy right skew and presence of outliers are typical of price data such as this and can cause some problems during estimation. As such, a log-transformation of the data was performed, which resulted in a much more symmetrical and "modelling-friendly" distribution.

<img src="images/Log_response_distribtion.png?raw=true"/>

### 2. Data Cleaning, Processing, and Wrangling

### 3. Feature Engineering

### 4. Modelling

### 5. Analysis

### 6. Appendix

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

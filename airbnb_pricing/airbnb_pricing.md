## A statistical machine learning based pricing recommendation system for Airbnb listings in Melbourne

**Project description:** This project was completed as an assessment piece for a machine learning and data mining unit at the University of Sydney. For this assignment, we were given a data set of AirBnB listings in Melbourne and were tasked with using statistical learning methods to build a pricing recommendation system. The data was given in two parts, a labelled train set and an unlabelled test set. The models were evaluated through Kaggle, where they were run on the complete, labelled test set. The project gave me a good opportunity to work with a data set that required a fair bit of data wrangling and feature engingeering and to then implement a variety of machine learning algorithms in Python.

A description of the train set can be found in the appendix.

### 1. Exploratory Data Analysis

The training set contains 7,000 listings and 28 independent variables as well as the target variable (price). The listing prices ranged from $13 per night to $5,000, with a mean of $152.19 and a median of $119.00. This implies a relatively significant right skew, which can be seen in distribution below.

<img src="images/Response_distribution.png?raw=true"/>

The heavy right skew and presence of outliers are typical of price data such as this and can cause some problems during estimation. As such, a log-transformation of the data was performed, which resulted in a much more symmetrical and "modelling-friendly" distribution. 

<img src="images/Log_response_distribution.png?raw=true"/>

Of the independent variables, six were categorical, and would thus require further feature engineering in order to be useable. Additonally, latitude and longitude offer no information in their current state; further processing would be required to derive useful insights. A pairwise comparison of each (numerical) independent variable with respect to the price/log price revealed that the majority of variables do not share a linear relationship with the target variable.

<img src="images/Pairwise_comparisons.png.png?raw=true"/>

With a number of very similar independent variables in the dataset, high multicollinearity was likely to pose a problem. The correlation between each feature was examined through a correlation matrix to better understand the underlying relationships that exist within the dataset.

<img src="images/Correlation_matrix.png?raw=true"/>

As seen above, the review scores display a relatively high positive correlation. This is logical, one would expect that a listing that is reviewed highly in one area would often also score highly in others. Additionally, review_score_rating is an aggregation of the rest of the scores; this implies perfect multicollinearity, which would need to be addressed through feature engineering. 

Based off the insights gained from the EDA, a fairly obvious hypothesis would be that due to the non-linearity and relatively high dimensionality of the data, linear models could struggle to capture the various relationships between the variables and would therefore perform poorly during prediction. Instead, non-linear methods would likely be superior in this case. The hypothesis is challenged and tested during the model estimation process.

### 2. Data Cleaning, Processing, and Wrangling

As seen in the dataset description, there were a number of variables with missing values. To remove the problem of missing data, observations with missing values could obviously be merely removed from the dataset. However, unless the observation is so sparsely populated that it doesn't offer much information, it's generally not ideal to delete data. Instead, the missing values could be imputed with another value.

There are a number of ways to approach data imputation. As a basic level, you can just replace the missing values with some measure of central tendency (mean, median, etc.). For time series data, something like linear interpolation is popular. Then there is also the possibilty to create a model to predict the missing values, and then use these predicted values in the estimation of the target variable.

In our case, there are three variables (bedrooms, bathrooms, and beds) that have a trival number of missing values (<7). Here, it makes sense just to impute the missing values with the respective median. That leaves ten variables with missing values - security_deposit, cleaning_fee, reviews_per_month and each of the seven review metrics - each with more than 1,000. Given the large number of missing values, filling them all with the same value is less likely to produce good predictions. However, we can try to predict them instead, K Nearest Neighbours (KNN) is a popular method for data imputation due to ease of implementation and relatively good results.

The underlying rationale for KNN imputation is that a point value can be approximated by using the values of the points that are closest to it, based on the other features within the data set (excluding the features that are missing large portions of data themselves). This method only requires the specification of the number of nearest neighbours (a ‘k’ value), and a distance metric. In the first instance, 5-fold cross-validation was used (taking computational cost into consideration) to determine an optimal k value of 17. Another important consideration is that due to the sheer volume of missing data entries, and the high dimensionality of the data, many missing values would not have enough neighbours to deliver reliable predictions. To account for this, if more than 50% of the nearest neighbours had missing values in the column of interest, the observation was left unfilled. 

The first run of KNN imputation filled the vast majority of missing values (e.g. 1,000 in the case of security_deposit). With many more values now populated, the process was repeated. However, after two iterations the algorithm the effectiveness diminished so it made sense just to stop it there so as to not compound any estimation bias. As there were only a relatively small number of missing data points remaining, they were filled using the median.

### 3. Feature Engineering  

The first candidates for feature engineering are the latitude and longitude columns. Coordinate values are obviously not of much use in the modelling process as the numbers themselves. But they do contain information that could be extracted for a range of useful predictors. To try and determine what kind of useful features could be created, I generated a heat map of the listings.

<img src="images/Zoomed_map.png?raw=true"/>

Here we can see that high-priced listings are clustered around each other, most notably in the CBD area, the inner north, and the inner east. This is no surprise, as every city in the world has higher priced and lower priced areas. Therefore, it seems logical that a listing's location may offer some explanatory power in predicting its price. The most well-defined definition of an area in Australian cities is a postcode. So therefore, I used the geopy module in Python to match the coordinates with the corresponding postcode. 

This resulted in almost 250 unique postcodes among the listings, 235 in the test set and 234 in the train set. While postcodes are numbers, their values have no inherent meaning and instead act as categories. As such, the feature needed further processing before it could be used for modelling. One option would be to create a dummy variable for each postcode (bar one). However, this would drastically increase the dimensionality of the dataset. Instead, I used median encoding, whereby the median of price is calculated for listings in each postcode and this median value then is placed into a single feature. The median was chosen over the mean so as to not be distorted by outliers. This process was simple for the train set but as the test set contained 14 postcodes not present in the train set, median prices are not available for these postcodes. The median price of the entire dataset is used instead for these observations.

Another important factor regarding location in a city, is not only the postcode but also the location relative to areas of interest.  Drawing upon my previous knowledge of the city, some basic logic, and the insights from the EDA, I picked out five locations: the CBD, the MCG/Melbourne Sports Precinct, Chapel Street, St Kilda, and Fitzroy. These five areas contains the city's most visited tourist attractions, beach areas, and nightlife districts. After deciding on the locations I needed a distance metric. The Euclidean distance, which gives the distance as the crow flies, may seem like the obvious choice. But Melbourne has a grid-like road system so a straight line is probably not reflective of the true distance needed to travel there. The Manhattan distance, however, is perfect for this. After measuring the distance from each location for each observation, I now had five more features that are much more useful that mere coordinate values. However, the five features are (as expected) highly correlated, all with 𝜌>0.9. To work around this, principal components analysis (PCA) was used. As seen on the chart below, the first principal component captured more than 97.5% and this became the feature for the final dataset, replacing the distances from the five individual locations; the remaining principal components were not considered.

<img src="images/PCA_distance_scree_plot.png?raw=true"/>

The property_type variable, contained 30 distinct classes. Only 11 of these classes were proporationally greater than 0.05%, and these 11 classes account for around 99% of all listings in the training set. Given the infrequency of the remaining 19 classes, I grouped them into an 'Other' class. This transformation simplified the categorical variable in preparation for a median encoding to be applied, as was done above for the postcodes. Three categorical variables (host_is_superhost, host_identity_verified, and instant_bookable) are be binary, listed as either ‘t’ or ‘f’, and thus could be transformed into dummy variables. I then used median encoding on the remaining categorical variables (room_type and cancellation_policy), each of which had fewer than five classes. The final component of the feature engineering is to deal with the perfect multicollinearity of the review scores. PCA is again perfect for this task; first three principal components explained over 83% of the variance within the seven variables and became part of the final feature set. 

<img src="images/PCA_reviews_scree_plot.png?raw=true"/>

A final list of features can be seen in Appendix.

### 4. Modelling 

Although we estimated a wide range of different models, I'll only focus on the five most relevant/interesting for this problem. As mentioned in the EDA, due to the heavy right skew of the initial target variable, the log transformation of price was used as the target variable for the models that are discussed in the report. A skewed distribution is common for data concerning property/rental prices and models estimated on the log-response often perform better as a result. This was the case in our analysis. All of the models were also fit using the untransformed price variable and as a whole performed worse.

The five models considered were:
- ℓ1 regularisation (lasso regression)
- KNN regression using principal components as regressors
- Random forest
- Gradient boost
- Model stack of lasso, KNN/PCA, random forest, and gradient boost

**Lasso Regression**  

The lasso regression was primarly fit in order to challenge the hypothesis that linear models would struggle to model the non-linear relationships in the data. After using 15-fold cross validation (CV) and testing 151 different regularisation parameter (𝛼) values, logarithmically spaced, with the function *np.logspace(-15, -10, 151, base=2)*. The chart below shows the loss for the corresponding 𝛼 values:

<img src="images/Lasso_lambda.png?raw=true"/> 

At the optimal 𝛼 of 1.0627<sup>-4</sup>, the final CV RMSE was higher than that of the other models tested. This supports the hypothesis that linear models would struggle to estimate the non-linearity in the data.

Optimal 𝛼 | RMSE
:--------:|:---:
1.0627<sup>-4</sup> | 144.26

**KNN with PCA**

Non-parametric methods are generally quite flexible and can therefore often model non-linearity quite well. However, they do have drawbacks. Namely, they suffer from the "curse of dimensionality", whereby the amount of data needed for accurate estimation quickly becomes unrealistic in higher dimensions. With 23 features, this issue is definitely relevant for this dataset. In order to overcome this, PCA was performed on the whole set of features; this included the highly correlated features that had been reduced before use in other models, but not including the principal components that resulted from that. From the scree plot below, we can see a noticeable ‘elbow’, after the third principal component, after which the following components fail to explain much variance.

<img src="images/KNN_PCA_scree_plot.png?raw=true"/>

Using this rule of thumb, a KNN regression was fit on both the first 3 componenents. After cross validation, we can see the optimal number of neighbours: 

<img src="images/KNN_PCA_3.png?raw=true"/>

The optimal points were 𝐾=9 for the model fit on the first 3 components respectively. This model proved to be an improvement upon the lasso regression, with the final results as below:

Principal Components | Optimal K-neighbours | RMSE
:-------------------:|:--------------------:|:---:
3 | 9 | 135.38

**Random Forest**

A random forest is an aggregation of a large number of decision trees where each tree is fit on a different bootstrapped data set with a random subset of features. This works to decorrelate the individual trees and reduce the risk of overfitting.  As a non-linear model, it is also well suited to this data set. There are a few different hyperparameters to tune when fitting a random forest:
- The number of estimators (i.e. no. of trees in the forest)
- The minimum number of samples allowed in leaf (end) nodes
- The maximum number of features used in each tree

Each of the above hyperparameters was tested at different values when estimating the model. While hyperparameter tuning can of course always be more granular, the running time increases exponentially with added combinations and eventually becomes infeasible. The optimal hyperparameter values and RMSE were as follows:

Max. Features | Min. Samples | No. of Estimators | RMSE
:------------:|:------------:|:-----------------:|:---:|
11 | 1 | 200 | 129.46

**Gradient Boost**

Another tree-based method, gradient boosting, is also particularly well suited to non-linearity. it is also a powerful model that regularly features in winning Kaggle submissions. Gradient boosting is an iterative algorithm, where each successive model/tree effectively tries to fanotherix the errors of the previous one. More specifically, we used XGBoost for the added benefit of regularisation, again in an attempt to avoid overfitting. Again, there are multiple hyperparameters. Many are the same as in the random forest, but one in particular is very different. That is the learning rate, which controls the size of the steps in the gradient descent optimisation algorithm. A higher learning rate means faster optimisation but risks overshooting the optimal point and a lower learning rate is more thorough but runs the risk of being too slow to properly train the model. The resulting best parameters and RMSE score achieved utilising these parameter values are shown below:  

Learning Rate | No. of Estimators | Max. Tree Depth | Sub-sample Ratio | RMSE
:------------:|:-----------------:|:---------------:|:----------------:|:---:|
0.05 | 750 | 3 | 1.0 | 132.82

**Model Stack**

The final model is a stack of the previous four and connected with a linear regression meta model that places a weight on the predictions of each. Due to the computational cost, 5-fold cross validation is used here rather than 15-fold as before. The traditional thinking with model stacks is to combine a number of weak learners to create a strong learner. In this case, rather than combining every single possible model, the four constituent models were carefully selected based on their individual strengths and weaknesses to combine the flexibility of non-linear models with benefits of linear models and regularisation methods. This was an attempt to avoid overfitting, as combining learners that all predict in a similar way will lead to collinearity between the predictors in the meta model and could (at least partially) contribute to overfitting. Hence the focus  on including a range of models that each approach the problem in a slightly different way. This combination of linear and non-linear learners resulted in a strong performance during the cross validation.

Constituent Models | Meta Model | RMSE
:-----------------:|:----------:|:---:|
Lasso, KNN with PCA, Random Forest, XGBoost | Linear regression | 129.27

### 5. Model Evaluation and Analysis

Below is the final performance of each of the five models:

Model |	Train CV RMSE |	Test RMSE
------|:-------------:|:--------:
Lasso	| 144.26	| 189.90
KNN with PCA	| 135.38 |	142.30
Random Forest	| 129.46	| 121.46
XGBoost	| 132.82	| 111.26
Model Stack	| 129.27	| 115.17

Although the random forest and model stack slightly outperformed it during cross validation, XGBoost gave the best predictions on the test set. Unfortunately, models such as this can somtimes be somewhat difficult to interpret. However, an interesting aspect that we can look at is the feature importance. This allows us to see which variables had the greatest impact on listing prices, which is naturally a very important piece of information for AirBnB hosts. By identifying the property features that are most important to customers and most influential on the price, hosts can adjust their pricing strategies accordingly in order to maximise demand/interest as well as revenue.

<img src="images/XGB_feature_importance.png?raw=true"/>

Here we can see the feature importance according to the gain criterion, which measures the relative contribution of the feature to the model, i.e. its importance relative of each feature for prediction, relative to the rest of the features. There are a few interesting insights we can draw from this. 

The first, and most obvious, is that the type of room (e.g. shared room, private room, whole apartment, etc.) is by far the most influential factor on price. For listing owners, this means that by carefully adjusting this, there could be a chance to maximise the revenue gained from the listing. The second, is that (perhaps unsurprisingly) the number of bedrooms and bathrooms as well as the capacity and location of the property are the next most influential property features. For existing owners, this information may not be all that useful, as this can't really be changed without considerable expense. However, for prospective owners looking to invest, this gives an indication of the property features to look out for. The final insight is perhaps somewhat less expected. If we look at the chart, we can see that the review score principal components are among the bottom few, indicating that the review scores are some of the influential factors on price. This suggests that listing owners need not worry too much about the quality of reviews when it comes to price. However, reviews may be important in other aspects, e.g. influencing customers when deciding between two similarly priced properties. This information is outside of the scope of this pricing recommendation but nonetheless an important consideration.


### 6. Conclusion

So now we have created a system that can take the physical features and properties of an AirBnB listing and suggest the correct listing price. The gradient boosted model gave the best performance for the recommendation system. Additionally, we found out that the type of room is by far the biggest factor that influences price. Secondary to this are factors related to the size of the listing (capacity, no. of bedrooms and bathrooms) as well as the postcode that the dwelling is in; none of this is surprising. Possibly the most interesting insight is that review scores play almost no role in determining the price. With this information, listing owners cannot only determine the right price to offer their apartment/room at, they can also identify important factors to increase the potential revenue of their listing.


### 7. Appendix

**Train Data Set**

Variable |	Non-null Values |	Data Type
---------|:----------------:|:--------:
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

**Final Feature List**

Feature |	Data Type
--------|:--------:
Host_is_superhost | Integer (64)
Host_total_listings_count | Integer (64)
Host_identity_verified | Integer (64)
Accommodates | Integer (64)
Bathrooms | Float (64)
Bedrooms | Float (64)
Beds | Float (64)
Security_deposit | Float (64)
Cleaning_fee | Float (64)
Extra_people | Integer (64)
Minimum_nights | Integer (64)
Maximum_nights | Integer (64)
Number_of_reviews | Integer (64)
Instant_bookable | Integer (64)
Property_type | Float (64)
Postcode | Float (64)
Distance_feature | Float (64)
Room_type | Float (64)
Cancellation_policy | Float (64)
Reviews_per_month | Float (64)
Review_pc_1 | Float (64)
Review_pc_2 | Float (64)
Review_pc_3 | Float (64)




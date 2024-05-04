## Statistics/ML Notes

### Statistics

#### Confidence Interval
- "Based on the sample data we have, if we were to extend our findings to the entire population, we are X% confident that the true population value would fall within this range."

#### Univariate Analysis
- When you analyze data by looking at one variable at a time.

#### Bivariate Analysis
- When you analyze two variables together.

#### Multivariate analysis
- When you evaluate multiple variables (more than two) to identify any possible association among them.

#### Outliers
- An outlier is an observation that deviates significantly from the majority of the datapoints in a dataset. Outliers can be caused by various factors like errors and entry mistakes, rare but valid extreme values and anomalies or unusual events.

#### Normal Distribution
- Gaussian distribution/bell curve, is a probability distribution that is symmetric about the mean and follows a specific mathematical formula.
- A perfect normal distribution has the mean, meadian and model be equal and located at the center of the distribution.
- A perfect normal distribution is symmetric, with 50% of the data falling on each side of the mean.
- A perfect normal distribution data follows the 68-95-99.7 rule, meaning approximately 68% of the data falls within one standard deviation of the mean, 95% within two standard deviations, and 99.7% within three standard deviations.


#### Noise
- Noise is random, irrelevant or meaningless information that can interfere with the interpretation and analysis of actual data. Noise makes it more difficult to discern the true pattern or signal from data because it introduces variability or errors that don't provide useful insights.


#### Standard Deviation
- A measure of how dispered the data is in relation to the mean.
- A low standard deviation indicates data is clustered around the mean, while high deviation sugguest data points are spread out.



### ML Techniques & Terms

#### Precision
- Precision is a measure of the 'exactness' or the quality of positive identifications made by a model.
- That is, if a model detects every positive case, but also detects a bunch of negative cases as well, then precision still decreases.

#### Recall
- Ability for the model to find all relevant cases (positives). 
- That is, for all actual positive cases, how many did the model correctly detect as positive?
- For example, if there are 100 postive cases, and the model detects 200, but the model correctly detects all 100 of the true cases, then recall would still be 100%.





#### Ensembling
- Bagging (Bootstrap aggregating)

#### Measuring importance of Nomninal-Categorical vs. Continuous/Ordinal Features
  - Nominal-Categorical
    - Often measured by observing how different categories relate to changes in the target variable.
      - Chi-square test
        - Test if two categorical variables are independent or influencing the test statistic
      - ANOVA
  - Continuous/Ordinal
    - Done by looking at correlation coefficients or use models that can measure feature importance.
      - Linear Regression Coefficients.



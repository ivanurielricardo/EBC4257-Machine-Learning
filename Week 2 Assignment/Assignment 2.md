# Assignment 2 Machine Learning

### Problem 1

Why might we use ensemble learners?
- They could result in a lower error
- Less overfitting
- Variety of models

Because we have a binomial distribution, we can use the binomial 
formula in order to calculate the probability that an ensemble 
has an error. In this case, n = 5, k = 3, and p = 0.3. This
results in us obtaining an error of 0.1323 for the total
ensemble method. 

### Problem 2

If errors are now partially correlated, we can probably use a 
Poisson binomial distribution. These variables may not necessarily
be identically distributed. Correlation in the error terms suggests that there is additional
information in the data that has not been exploited in the current
model.

### Problem 3

The point of an ensemble method is to lessen the variance of a
high variance model. When we create an ensemble method, we 
increase the bias a tiny bit by averaging in order to obtain
a combination or "ensemble" of methods that contains less variance.

### Problem 4

What is the outcome for the CART algorithm for bagging? By bagging,
we don't use the entirety of the dataset on our training set. 
We use the bootstrap in order to create a forest of trees and 
combine the results from each in order to obtain an ensemble 
which has less variance than the original tree would have. 

### Problem 5

Receiver operator Characteristic: provides a simple way to summarize
all the information you obtain. X axis holds false positive rate
and Y axis holds the true positive rate (sensitivity 
vs specificity). The farther away you are from the dotted line, the 
decrease in the proportion of samples that were incorrectly 
classified as obese. 

In terms of goodness of fit, being farther away could also mean 
more potential to be overfit. 

### Problem 6

What is a learning curve?

This is the correlation between the learner's performance on a
task and the number of attempts or time required to complete the
task; this can be represented as a direct proportion on a graph.

Learner's efficiency in a task improves over time the more times
learning procedure is repeated.

### Problem 7

Programming

Feature 1 has an importance of 68% and feature 2 has an importance
of 31.1%.
Unfortunately did not do so for the second scenario :(

### Problem 8

Comparing the ROC curve and learning curves of the tree


### Will continue soon...
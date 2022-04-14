rm(list=ls()) # Remove all items within working memory
?sample
library(ISLR2)
set.seed(1)
train <- sample(392, 196)
######### Validation Set Approach #########

# use the subset option to fit a linear regression using only the observations 
# corresponding to the training set

lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train)
summary(lm.fit)

# use predict to estimate the response for all 392 observations
# also use mean to calculate MSE of 192 observations in the training set

attach(Auto)
MSE1 <- mean((mpg - predict(lm.fit, Auto))[-train]**2)

# can now try different models and test their performance

lm.fit2 <- lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
MSE2 <- mean((mpg - predict(lm.fit2, Auto))[-train]^2)

lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train)
MSE3 <- mean((mpg - predict(lm.fit3, Auto))[-train]^2)

# note that if we chose a different training set, the MSE would be different

######### BOOTSTRAP #########

# Estimating the accuracy of a statistic of interest (Standard Error)
library(boot)

# we now use a bootstrap to repeatedly sample observations with replacement

# create a function which takes inputs (X,Y) data and a vector indicating which
# observations should be used to estimate \alpha

alpha.fn <- function(data, index) {
  X <- data$X[index]
  Y <- data$Y[index]
  (var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2 * cov(X, Y))
}
# returns an estimate for alpha based on the equation found in 5.7 (ISLR)
# recall \alpha is the probability of obtaining X used as a baseline for the bootstrap

set.seed(7)
alpha.fn(Portfolio, sample(100, 100, replace = T))

# can now implement the bootstrap analysis by implementing this multiple times
# produce R = 1000 bootstrap estimates for \alpha (which can change)

boot(Portfolio, alpha.fn, R = 1000)
# using the original data and an \alpha level of 0.5758, our SE is 0.0897


# Estimate the accuracy of a linear regression model

# assess variability of the estimates for B_0 and B_1

boot.fn <- function(data, index) {
  coef(lm(mpg ~ horsepower, data = data, subset = index))
}

boot.fn(Auto, 1:392)

# this is an easier way to set up the lm function for a bootstrap

set.seed(1)

boot.fn(Auto, sample(392, 392, replace = T))
boot.fn(Auto, sample(392, 392, replace = T))

# two different estimates for the intercept and horsepower!
# use boot function to compute std. errors of 1000 bootstrap estimates for int. and slope

boot(Auto, boot.fn, R = 1000)

# bootstrap estimates for SE(B_0) is 0.84

summary(lm(mpg ~ horsepower, data = Auto))$coef

# standard errors are different - displays the problem with bootstraps
# if our model doesn't fit the data precisely, the bootstrap could lead to bad results

# fit with a quadratic

boot.fn <- function(data, index) {
  coef(lm(mpg ~ horsepower + I(horsepower**2), data = data, subset = index))
}

set.seed(1)

boot(Auto, boot.fn, 1000)





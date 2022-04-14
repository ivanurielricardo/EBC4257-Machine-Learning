rm(list=ls()) # Remove all items within working memory

library(MASS)
library(ISLR2)


################## Linear Regressions ##################
head(Boston)

# Need to predict median value using 12 predictors such as rm, age, lstat
lm.fit <- lm(medv ~ lstat, data = Boston) 
summary(lm.fit)
confint(lm.fit)

# the predict function can be used in order to produce confidence intervals and prediction intervals
# just need to switch confidence and prediction

predict(lm.fit, data.frame(lstat = (c(5,10,15))),interval = "prediction")

plot(lstat, medv, pch = 10)
abline(lm.fit, lwd = 1, col = "red")

par(mfrow = c(1,1))
# plot(lm.fit)

# computing the residuals may come in handy
# function rstudent returns studentized residuals

plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))

plot(hatvalues(lm.fit))
which.max(hatvalues((lm.fit))) # tells us the observation with the highest leverage

################## Multiple Regressions and Interaction Terms ##################

lm.fit2 <- lm(medv ~ . - age - indus, data = Boston) # include all variables into regression
summary(lm.fit2)
plot(lstat, medv)
abline(lm.fit2)

summary(lm(medv ~ lstat * age))

################## Nonlinear Transformations ##################

lm.fit3 <- lm(medv ~ lstat + I(lstat^2))
summary(lm.fit3)

plot(lstat, medv)
abline(lm.fit3, col = "red")

anova(lm.fit, lm.fit3)
# null hypothesis is that the two models fit the data equally well. 
# an F-stat of 135 implies that second model is superior

par(mfrow = c(1,1))
plot(lstat, medv)
abline(lm.fit3, col = "red")
abline(lm.fit2, col = "blue")

# red is still a bit wonky but a better fit


lm.fit4 <- lm(medv ~ poly(lstat, 5))
summary(lm.fit4)

# anova wouldn't work when we have log transform on response variable

anova(lm.fit3, lm.fit4)

# still prefer the fourth model




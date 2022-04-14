####### Classification Tree #######
## Gathering the Data
rm(list=ls()) # Remove all items within working memory


library(tree)
library(ISLR2)
attach(Carseats)


head(Carseats)
# we have a variety of categorical data and numerical data
?Carseats

high <- factor(ifelse(Sales <= 8, "No", "Yes"))
Carseats <- data.frame(Carseats, high)
head(Carseats)
# New high variable takes "yes" if sale value exceeds 8, no otherwise

# we can use the tree function to create a classification tree

tree.carseats <- tree(high ~ . - Sales, Carseats)
summary(tree.carseats)
# misclassification error rate is 0.09 for training data

# we can plot this decision tree
plot(tree.carseats)
text(tree.carseats, pretty = 0, cex = 0.6)

# Most important indicator of sales seems to be shelve location, based on the root node
# next most important is price, split between 92 and 135

# performing cross validation
set.seed(2)
train <- sample(1:nrow(Carseats), 200)
# taking the first 200 samples
carseats.test <- Carseats[-train,]
high.test <- high[-train]
tree.carseats <- tree(high ~ . - Sales, Carseats, subset = train)
tree.predict <- predict(tree.carseats, carseats.test, type = "class")
table(tree.predict, high.test)
(104 + 50)/200
# confusion matrix
# we correctly classify 77% of the time

####### Regression Tree #######

set.seed(1)
?plot
head(Boston)
# here we have numerical data
df.Boston <- data.frame(Boston)
plot(df.Boston[,3], df.Boston[,7], data = df.Boston, xlab = "INDUS", ylab = "AGE",
     pch = 20)



train <- sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston <- tree(medv ~ age + indus, data = Boston, subset = train)
summary(tree.boston)

plot(tree.boston)
partition.tree(tree.boston, col = "black", add =  TRUE)
text(tree.boston, pretty = 0, cex = 0.6)
# most important is rooms per dwelling for the median value
# next most important is the lower status of the population
# we could have fit a much larger tree, but can cross validate to determine best model

cv.boston <- cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = 'b')
# would pruning the tree improve performance?
# The most complex tree is chosen by cross validation, but can prune the tree regardless

prune.boston <- prune.tree(tree.boston, best = 5)
plot(prune.boston)
text(prune.boston, cex = 0.6)

# keeping with CV, we use unpruned to make predictions on the test set
yhat <- predict(tree.boston, newdata = Boston[-train,])
boston.test <- Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0,1, col = "blue")
mean((yhat-boston.test)**2)
# the test set MSE is 35.28688


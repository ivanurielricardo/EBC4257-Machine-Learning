############# Problem Sets ############# 

rm(list=ls()) # Remove all items within working memory

library(tree)
library(ISLR2)
attach(Carseats)
library(kernlab)
library(tidyverse)

set.seed(3)


############# Problem 6 #############


# We seek to predict Sales using a Regression tree, treating the response as quantitative

train <-  sample(dim(Carseats)[1], dim(Carseats)[1]/2)
Carseats.train <-  Carseats[train, ]
Carseats.test <-  Carseats[-train, ]

# We fit a regression tree onto the training set

tree.Carseats <- tree(Sales ~ ., Carseats.train)
summary(tree.Carseats)

plot(tree.Carseats)
text(tree.Carseats, pretty = 0, cex = 0.7)

# Here, the most important indicator of Sales seems to be the quality of shelf location
# The next most important indicator is the price 

# We follow the tree given the questions and we can then make an approximation of how
# many sales a store could sell

# What if we were to change the seed of the seed of the random number generator?

# seed = 3: (benchmark)
# seed = 4: average age of pop becomes more important than price
# seed = 5: price is now more important than age of carseat. shelf location is still most important
# seed = 6: nothing interesting. Interesting note is branches are also stable
# seed = 7: again nothing interesting. Tree stays stable


############# Problem 9 #############

# Let us now make predictions on the test set

yhat <- predict(tree.Carseats, Carseats.test)
plot(yhat, Carseats.test$Sales)
abline(0,1, col = "blue")
mean((yhat-Carseats.test$Sales)**2)

# Our test MSE is 4.78


# would pruning the tree give better results? We can check using cross-validation
cv.Carseats <- cv.tree(tree.Carseats)
plot(cv.Carseats$size, cv.Carseats$dev, type = "b")

crossval <- data.frame(cv.Carseats$size, cv.Carseats$dev)

# TODO find argmin of a point in a dataframe

# In this case, we could prune the tree to 5 as it results in the lowest deviance

prune.Carseats <- prune.tree(tree.Carseats, best = 5)
plot(prune.Carseats)
text(prune.Carseats, cex = 0.6)
summary(prune.Carseats)

# Does pruning the tree improve MSE?

prune.yhat <- predict(prune.Carseats, newdata = Carseats.test)
plot(prune.yhat, Carseats.test$Sales)
abline(0,1, col = "blue")
mean((prune.yhat-Carseats.test$Sales)**2)

# We obtain an MSE of 5.39! Higher than the unpruned tree

# Confusion Matrix for the Pruned and Unpruned trees


# Here, we need to create a factor in order for the confusion matrix to work.
# We will create a factor of sales which is "High" if sales of carseats are more than 7.5 (Mean)

High <- factor(ifelse(Sales<=7.5, "No", "Yes"))
Carseats <- data.frame(Carseats, High)

carseats.test2 <- Carseats[-train,]
High.test <- High[-train]
tree.carseats2 <- tree(High ~ . - Sales, Carseats, subset = train)
tree.predict <- predict(tree.carseats2, carseats.test2, type = "class")
table(tree.predict, High.test)
# Normal Confusion matrix for unpruned tree

prune.Carseats2 <- prune.tree(tree.carseats2, best = 5)
prune.predict <- predict(prune.Carseats2, carseats.test2, type = "class")
table(prune.predict, High.test)
# Confusion matrix for pruned tree. Which did better?

# Sensitivity vs Specificity

############# Problem 10 ############# 

# what happens when we use k-fold cross validation?

cv.train <- sample (1: nrow(Carseats), nrow(Carseats)/2)
cv.tree.carseats <- tree(Sales ~ ., data = Carseats, subset = cv.train)
cv.tree(cv.tree.carseats, K = 10)

plot(cv.tree.carseats)
text(cv.tree.carseats)

############# Problem 11 ############# 

# We use the Spam dataset from the kernlab library
data(spam)
# each value indicates the percentage of words that appear in the email

# spam <- spam %>% 
#   mutate(type = recode(type, 
#                     "spam" = "1", 
#                     "nonspam" = "0"))
# Convert the "type" column into a binary variable

train <- sample(dim(spam)[1], dim(spam)[1]/2)
spam.train <- spam[train,]
spam.test <- spam[-train,]



tree.spam <- tree(type ~ ., data = spam, subset = train)
summary(tree.spam)

plot(tree.spam)
text(tree.spam, pretty = 0, cex = 0.6)

pred.spam <- predict(tree.spam, spam.test)
table(pred.spam, spam.test)





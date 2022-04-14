############# Problem Sets ############# 

rm(list=ls()) # Remove all items within working memory

library(tree)
library(ISLR2)
library(tidyverse)
library(kernlab)

set.seed(3)


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

# Now we can try making predictions on the tree

spam.yhat <- predict(tree.spam, spam.test, type = "class")
plot(spam.yhat, spam.test$type)
table(spam.yhat, spam.test$type)

sensitivity <- 752/(752+149)
specificity <- 1299/(1299+101)
# Sensitivity is 0.8346 and Specificity is 0.9278

# We can compare this with a pruned tree



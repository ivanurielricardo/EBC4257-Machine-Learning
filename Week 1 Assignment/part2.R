############# Problem Sets ############# 

rm(list=ls()) # Remove all items within working memory

library(tree)
library(ISLR2)
library(tidyverse)
library(kernlab)


set.seed(3)


############# Problem 12 ############# 

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
spam.confusion <- table(spam.yhat, spam.test$type)
spam.confusion

sensitivity <- spam.confusion[2,2]/sum(spam.confusion[,2])
specificity <- spam.confusion[1,1]/sum(spam.confusion[,1])
# Sensitivity is 0.8324 and Specificity is 0.9591

# We can use cross validation to check the pruned trees

cv.spam <- cv.tree(tree.spam)
plot(cv.spam$size, cv.spam$dev, type = "b")

# Pruning the tree would yield no better results

# Lets assume that pruning the tree would yield better results


############# Problem 13 ############# 

# We now find the MNIST dataset


devtools::install_github("rstudio/keras") 
# installing keras 
# It will first install tensorflow then keras.
# The above code will install the keras library from the GitHub repository.

install.packages("rstudio/keras")

#loading keras in R 
library(keras)
#The R interface to Keras uses TensorFlow as it’s underlying computation engine.

#loading the keras inbuilt mnist dataset
data<-dataset_mnist()

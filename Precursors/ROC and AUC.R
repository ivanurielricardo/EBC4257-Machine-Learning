rm(list=ls()) # Remove all items within working memory

library(pROC)
library(randomForest)

set.seed(420)

# We will use an example data set from the StatQuest AUC and ROC
# Our examples are based on OBESE people - how do we classify?

num.samples <- 100

weight <- sort(rnorm(num.samples, mean=172, sd=29)) # create our data set
# Average man weighs 172 wih a sd of 29!
# sort is used to sort from low to high

obese <- ifelse(test=(runif(n=num.samples) < (rank(weight)/100)), yes = 1, no = 0)
# classification as obese or not obese
# rank weights from lightest to heaviest
# lightest sample has rank = 1, heaviest rank = 100
# scale the ranks by 100
# this is used to normalize through the uniform distribution
obese

plot(x=weight, y=obese)

glm.fit <- glm(obese ~ weight, family = "binomial")
lines(weight, glm.fit$fitted.values)
# glm.fit$fitted.values contains estimated probabilities that each sample is obese

# use the ROC function to create a ROC graph 
roc(obese, glm.fit$fitted.values, plot = TRUE)

# As can be seen, we have true positives on the y axis and false positives on the x axis
# Here, we have 45 samples that were not obese and 55 that were obese
# AUC is 0.8291
# 1 - specificity is on the x axis - this means the same for our purposes

roc(obese, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE)
# Can make this easier to read

roc(obese, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive Percent", ylab = "True Positive Percent", col = "red", lwd = 4)

# imagine we were interested in the range of thresholds that resulted in a specific portion of the ROC

roc.info <- roc(obese, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE)
roc.df <- data.frame(tpp=roc.info$sensitivities*100, fpp = (1-roc.info$specificities)*100,
                     thresholds=roc.info$thresholds)
# this is a df that contains all true positive, false positives, and thresholds

head(roc.df)
# when threshold = -inf, then tpp is 100 because all obese samples were correctly specified
# this corresponds to the top right corner of ROC

# we want to isolate tpp and fpp (true positive/false positive) tpp is between 60 & 80
roc.df[roc.df$tpp>60 & roc.df$tpp<80,]

# we could pick a threshold within this matrix

roc(obese, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive Percent", ylab = "True Positive Percent", col = "red", lwd = 4,
    print.auc=TRUE)

# can also calculate a partial AUC
# useful if you only want to focus on part of ROC that allows for a small number of fpp

roc(obese, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive Percent", ylab = "True Positive Percent", col = "red", lwd = 4,
    print.auc=TRUE, print.auc.x = 65, partial.auc = c(100,80), auc.polygon = TRUE, 
    auc.polygon.col = "blue")

# partial.auc denotes the range of specificity we want to focus on

# Say we want to overlap 2 ROC curves in order to compare different models

rf.model <- randomForest(factor(obese) ~ weight)

roc(obese, glm.fit$fitted.values, plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive Percent", ylab = "True Positive Percent", col = "red", lwd = 2,
    print.auc=TRUE)
plot.roc(obese, rf.model$votes[,1], percent=TRUE, col = "blue", lwd = 2, print.auc = TRUE, add = TRUE,
         print.auc.y = 40)
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col = c("red", "blue"),
       lwd = 4, cex = 0.6)
# based on the AUC, the logistic curve is better than the random forest when it comes to correct
# specification
# for random forest, we pass the number of trees in the forest that voted correctly
# Example of scenarios 1 and 2 as described in Elements of Statistical Learning II (page 13), no warranty.
# Course: EBC4257 - Machine Learning, SBE, UM
# Author: RJAlmeida, April 2021
rm(list=ls()) # Remove all items within working memory


mu1=c(0, 1)
mu2=c(2, 3)
n_data_points = 1000

library(MASS)
library(tree)

set.seed(42)

scenarios_generation <- function(mu1=c(0, 1), mu2=c(2, 3), n_data_points = 1000) {
  # Scenario 1
  x1 = mvrnorm(n_data_points/2, mu = mu1, Sigma = diag(2))
  x2 = mvrnorm(n_data_points/2, mu = mu2, Sigma = diag(2))
  y1 <- rep("Yes", n_data_points/2)
  y2 <- rep("No", n_data_points/2)
  
  scenario1.y <- as.factor(c(y1, y2))
  scenario1.X <- rbind(x1, x2)
  
  data_scenario1 <- data.frame(scenario1.y, scenario1.X)
  colnames(data_scenario1) <- c("Y", "X1", "X2")
  
  # plots
  cols = 3-ifelse(data_scenario1[,1]== "Yes",1, -1)
  plot(data_scenario1[,-1], col = cols, main = "Scenario 1")
  abline(lm(scenario1.y ~ scenario1.X))
  legend('topleft', c('Yes', 'No'), pch = 1, col = c(2, 4))
  
  # training set
  # train.1 <- sample(1:nrow(data_scenario1), nrow(data_scenario1)/2)
  
  # tree generation
  # tree.scen1 <- tree(scenario1.y ~ scenario1.X, data = data_scenario1, subset = train.1)
  
  # new plots
  # plot(tree.scen1)
  # text(tree.scen1, pretty = 0, cex = 0.5)
  
  # Scenario 2
  mean_matrix <- mvrnorm(10, mu = mu1, Sigma = diag(2))
  mean_matrix_2 <- mvrnorm(10, mu = mu2, Sigma = diag(2))
  data_yes <- c()
  data_no <- c()
  for (kk in 1:10) {
    aux_yes <- mvrnorm(n_data_points/10, mu = mean_matrix[kk,], Sigma = .1*diag(2))
    data_yes <- rbind(data_yes, aux_yes)
    
    aux_no <- mvrnorm(n_data_points/10, mu = mean_matrix_2[kk,], Sigma = .1*diag(2))
    data_no <- rbind(data_no, aux_no)
  }
  
  y_yes <- rep("Yes", n_data_points/2)
  y_no <- rep("No", n_data_points/2)
  
  scenario2.y <- as.factor(c(y_yes, y_no))
  scenario2.X <- rbind(data_yes, data_no)
  
  data_scenario2 <- data.frame(scenario2.y, scenario2.X)
  colnames(data_scenario2) <- c("Y", "X1", "X2")
  
  #plots
  cols = 3-ifelse(data_scenario2[,1]== "Yes",1, -1)
  plot(data_scenario2[,-1], col = cols, main = "Scenario 2")
  abline(lm(scenario2.y ~ data_yes + data_no, data_scenario2))
  legend('topleft', c('Yes', 'No'), pch = 1, col = c(2, 4))

  return(list(data_scenario1, data_scenario2))
}

data_scenarios <- scenarios_generation()

summary(lm(scenario2.y ~ data_yes + data_no, data_scenario2))

########### Actual ###########
rm(list=ls()) # Remove all items within working memory

misclassification.error <- function(p) {
  1 - pmax(p, 1-p)
}

cross.entropy <- function(p) {
  -p * log2(p) - (1-p) * log2(1-p)
}

gini.index <- function(p) {
  2*p*(1-p)
}

curve(cross.entropy(x), 0, 1, col = "violet", xlab = "p",
      ylab = "Impurity")
curve(gini.index(x), 0, 1, add = TRUE,col = "blue")  
curve(misclassification.error(x), 0, 1, add = TRUE, col = "green")


legend("bottomright", legend = c("Cross-Entropy", "Gini", "Misclassification"),
       col = c("violet", "blue", "green"),
       lwd = 2, cex = 0.4)

scaled.cross.entropy <- function(p) {
  cross.entropy(p) / (2 * cross.entropy(0.5))
}

curve(scaled.cross.entropy(x), 0, 1, add = TRUE, col = "red")

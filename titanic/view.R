library(ggplot2)
library(psych)
library(reshape2)
library(caTools)

setwd("M:/users/matheus/crawler/ml/titanic")

train = read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)

factor(train$Embarked, order=TRUE)
describe(train)
str(train)
#Age has missing values
unique(train$Embarked)
train = subset(train, Embarked != "")
write.csv(train, file = "train_complete.csv",row.names=TRUE)

train$numSex[train$Sex == "male"] <- 1
train$numSex[train$Sex == "female"] <- 0

ggplot(data = melt(train), mapping = aes(x = value)) + 
  geom_histogram(bins = 50) + facet_wrap(~variable, scales = 'free')

train$PassengerId <- NULL

train[is.na(train[,5]), 5] <- mean(train[,5], na.rm = TRUE)

train$Name <- NULL
train$Sex <- NULL
train$Cabin <- NULL
train$Embarked <- NULL
train$Ticket <- NULL
cor(train, use="complete.obs",method = "pearson")


library(gclus)
#dta <- house[c(9,8,4,3)] # get data
dta <- train
dta.r <- abs(cor(dta)) # get correlations
dta.col <- dmat.color(dta.r) # get colors
# reorder variables so those with highest correlation
# are closest to the diagonal
dta.o <- order.single(dta.r) 
cpairs(dta, dta.o, panel.colors=dta.col, gap=.5,
       main="Variables Ordered and Colored by Correlation" )

train = read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
train = subset(train, Embarked != "")

train_predicted = read.csv("gender_submission_train.csv", header = TRUE, stringsAsFactors = FALSE)
train$predicted <- train_predicted$Predicted
subset(train, predicted != Survived)

measurePrecisionRecall <- function(predict, actual_labels){
  precision <- sum(predict & actual_labels) / sum(predict)
  recall <- sum(predict & actual_labels) / sum(actual_labels)
  fmeasure <- 2 * precision * recall / (precision + recall)
  
  cat('precision:  ')
  cat(precision * 100)
  cat('%')
  cat('\n')
  
  cat('recall:     ')
  cat(recall * 100)
  cat('%')
  cat('\n')
  
  cat('f-measure:  ')
  cat(fmeasure * 100)
  cat('%')
  cat('\n')
}

measurePrecisionRecall(train$predicted, train$Survived)

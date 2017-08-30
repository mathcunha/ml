setwd("D:/VMS/debian/ml/houseprice")

library(ggplot2)
library(psych)
library(reshape2)
library(caTools)

data= read.csv("train.csv", header = TRUE)

ggplot(data = melt(data), mapping = aes(x = value)) + 
  geom_histogram(bins = 50) + facet_wrap(~variable, scales = 'free')
str(data)
sapply(data, is.numeric)
describe(data)
size = 1460


unique(data["MasVnrType"])


setwd("M:/users/matheus/crawler/ml/chap2")
house= read.csv("housing.csv", header = TRUE, stringsAsFactors = FALSE)

library(ggplot2)
library(rworldmap)
library(rgdal)
library(mapproj)
library(psych)
library(reshape2)

newmap <- getMap(resolution = "low")
plot(newmap, xlim = c(-120, -110), ylim = c(30, 40), asp = 1)
points(house$lon, house$lat, col = "red", cex = .6)

ggplot() +  
  geom_polygon(data=counties, aes(x=long, y=lat, group=group))+
  geom_point(data=house, aes(x=longitude, y=latitude), color="red")

states<-readOGR("cb_2016_us_state_20m.shp")
counties<-readOGR("CA_counties.shp")

house$fMedianHouseValue <- cut(house$median_house_value,8)
house$fMedianIncome <- cut(house$median_income,6)
house$fPopulation <- cut(house$population,5)
describe(house)
summarise(house)
ggplot(data = melt(house), mapping = aes(x = value)) + 
  geom_histogram(bins = 50) + facet_wrap(~variable, scales = 'free')


set.seed(42) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample  <- sample.int(n = nrow(house), size = floor(.75*nrow(house)), replace = F)
train <- house[sample, ]
test  <- house[-sample, ]

require(caTools)

house$income_cat <- ceiling(house$median_income/1.5)
house$income_cat[house$income_cat > 5.0] <- 5.0

ggplot(data=house, aes(x=income_cat))+
  geom_bar(stat="count")+
  geom_text(stat="count",vjust=1, aes(y=..count.., label=formatC(..count../sum(..count..), 6, format="f") ))

set.seed(42) 
sample = sample.split(house$income_cat, SplitRatio = .2)
train = subset(house, sample == TRUE)
test  = subset(house, sample == FALSE)

ggplot(data=train, aes(x=income_cat))+
  geom_bar(stat="count")+
  geom_text(stat="count",vjust=1, aes(y=..count.., label=formatC(..count../sum(..count..), 6, format="f") ))

house$income_cat <- NULL
train$income_cat <- NULL
test$income_cat <- NULL

train$fMedianHouseValue <- cut(train$median_house_value,8)
train$fMedianIncome <- cut(train$median_income,6)
train$fPopulation <- cut(train$population,5)

ggplot() +  
  geom_polygon(data=counties, aes(x=long, y=lat, group=group), fill = "white", color ="black")+
  coord_cartesian(xlim = c(-125, -112), ylim = c(31, 42))+
  geom_point(data=train, aes(x=longitude, y=latitude, alpha=0.4, color=fMedianHouseValue, size=fPopulation))+
  scale_color_brewer(palette = "RdBu",direction = -1)

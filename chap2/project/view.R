library(ggplot2)
library(psych)
library(reshape2)
library(caTools)

setwd("D:/VMS/debian/ml/chap2/project")
house= read.csv("housing.csv", header = TRUE, stringsAsFactors = FALSE)


describe(house)
ggplot(data = melt(house), mapping = aes(x = value)) + 
  geom_histogram(bins = 50) + facet_wrap(~variable, scales = 'free')

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


write.csv(train, file = "train.csv",row.names=TRUE)
write.csv(test, file = "test.csv",row.names=TRUE)

train$fMedianHouseValue <- cut(train$median_house_value,8)
train$fMedianIncome <- cut(train$median_income,6)
train$fPopulation <- cut(train$population,5)

library(rgdal)
counties<-readOGR("CA_counties.shp")
ggplot() +  
  geom_polygon(data=counties, aes(x=long, y=lat, group=group), fill = "white", color ="black")+
  coord_cartesian(xlim = c(-125, -112), ylim = c(31, 42))+
  geom_point(data=train, aes(x=longitude, y=latitude, alpha=0.4, color=fMedianHouseValue, size=fPopulation))+
  scale_color_brewer(palette = "RdBu",direction = -1)

house$fMedianHouseValue <- NULL
house$fMedianIncome <- NULL
house$fPopulation <- NULL
house$ocean_proximity <- NULL

cor(house,method = "pearson")
library(gclus)
dta <- house[c(9,8,4,3)] # get data
dta.r <- abs(cor(dta)) # get correlations
dta.col <- dmat.color(dta.r) # get colors
# reorder variables so those with highest correlation
# are closest to the diagonal
dta.o <- order.single(dta.r) 
cpairs(dta, dta.o, panel.colors=dta.col, gap=.5,
       main="Variables Ordered and Colored by Correlation" )


house$rooms_per_household = house$total_rooms / house$households
house$bedrooms_per_room = house$total_bedrooms / house$total_rooms
house$population_per_household = house$population / house$households
cor(house, use="complete.obs",method = "pearson")


housing= read.csv("train.csv", header = TRUE, stringsAsFactors = FALSE)
housing_labels <- data.frame(housing$median_house_value)
names(housing_labels) <- c("median_house_value")
housing$median_house_value <- NULL

housing[is.na(housing[,6]), 6] <- mean(housing[,6], na.rm = TRUE)

for(i in 1:ncol(data)){
  data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
}

#NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))
housing$ocean_proximity <- NULL
describe(housing)
#replace(housing, TRUE, lapply(housing, NA2mean))
write.csv(housing, file = "train_complete.csv",row.names=TRUE)

lm(housing_labels$median_house_value ~ housing)
lm(housing_labels$median_house_value~.,housing)

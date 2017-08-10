setwd("M:/users/matheus/crawler/ml/chap2")
house= read.csv("cal_housing.csv", header = TRUE, stringsAsFactors = FALSE)

library(ggplot2)
library(rworldmap)
library(rgdal)
library(mapproj)

newmap <- getMap(resolution = "low")
plot(newmap, xlim = c(-120, -110), ylim = c(30, 40), asp = 1)
points(house$lon, house$lat, col = "red", cex = .6)

ggplot() +  
  geom_polygon(data=counties, aes(x=long, y=lat, group=group))+
  geom_point(data=house, aes(x=longitude, y=latitude), color="red")

states<-readOGR("cb_2016_us_state_20m.shp")

house$fMedianHouseValue <- cut(house$medianHouseValue,8)
house$fMedianIncome <- cut(house$medianIncome,6)
str(house)

ggplot() +  
  geom_polygon(data=states, aes(x=long, y=lat, group=group), fill = "white", color ="black")+
  coord_cartesian(xlim = c(-125, -112), ylim = c(31, 42))+
  geom_point(data=house, aes(x=longitude, y=latitude, color=fMedianHouseValue))+
  scale_color_brewer(palette = "YlOrRd")

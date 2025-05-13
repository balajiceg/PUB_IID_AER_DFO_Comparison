library(spdep)
library(sf)

tracts_sf <- st_read('PATH_TO_SHAPE_FILE')
neighbours = poly2nb(tracts_sf,row.names = tracts_sf$OBJECTID_12)

out_df <- data.frame(OBJECTID_12= tracts_sf$OBJECTID_12)
out_df$neighbours_OBJECTID_12<- lapply(neighbours, paste0,collapse=',') %>% unlist
write.csv(out_df,'censusTracts_AER_DFO_flood_neighbours.csv')

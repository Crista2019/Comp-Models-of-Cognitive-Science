#### Setup ####
# clear workspace
rm(list=ls(all=TRUE))

#### Packages ####
# package for plotting
library(ggplot2)
# package for data frames with benefits
library(data.table)
# package for piping
library(dplyr)
# regex
library(tidyverse)
library(tidyr)
library(plyr)
# for multipanel
library(patchwork)

#### Settings ####
# create an empty list
settings <- list()
# store path to workshop folder
# settings$path <- '~/tufts' # personal computer
settings$path <- '~/Documents' # lab computer

###### import data ######
# set current working directory as the folder for the localization task
setwd(paste0(settings$path, '/Lab/FYP/Data/Prepared/Localization')) # lab computer

# list all txt files in data directory
all_files <- list.files(pattern="")
# read them into a list of data.tables
dt.localization <- rbindlist(lapply(all_files, function(filename){
  # read file into data table
  tmp <- fread(filename)
  # add conditions to data table
  # grabs `Localization` from the file name
  tmp[, task := unlist(strsplit(filename,'-'))[1]]
  # truncates `1_Left_Shoulder.csv` to `Left_Shoulder` for example
  tmp[, area := substr(unlist(strsplit(filename,'-'))[3],3,(nchar(unlist(strsplit(filename,'-'))[3])-4))]
  tmp[, handed_area := ifelse(area=="Right_Shoulder", "dominant", ifelse(area=="Left_Shoulder", "nondominant", "center")) ]
  }), 
fill = T)

dt.localization[participant=="S19" & area=="Left_Shoulder"]$handed_area = "dominant"
dt.localization[participant=="S19" & area=="Right_Shoulder"]$handed_area = "nondominant"


# check trials without a response
dt.localization[is.na(responsePx)]
dt.localization <- dt.localization[!is.na(responsePx)]
# replace empty responses for confidence
dt.localization[confidence =='', confidence := NA]

#### recoding ####
# recode confidence ratings into numerical variable
dt.localization[, confidence.num := ifelse(confidence == 'high', 1, 
                                           ifelse(confidence == 'low', 0, NA))]

###### making factors for area and accuracy labels #####
oldAreas <- c("Left_Shoulder", "Midline", "Right_Shoulder")
newAreas <- factor(c("Left Shoulder", "Body Midline", "Right Shoulder"))

oldAcc <- c("FALSE","TRUE")
newAcc <- factor(c("Incorrect","Correct"))

# make a conversion function to map binned # units into real mm values
convertToMm <- function(vals) {
  # ((val_a - min_a)/(max_a - min_a))*(max_b-min_b)+min_b
  round(((vals - 1)/(12 - 1))*(329 + 76)-76)
}

convertToBins <- function(vals) {
  # ((val_a - min_a)/(max_a - min_a))*(max_b-min_b)+min_b
  (vals + 76)/(329 + 76)*(12-1)+1
}

# binning the location of trial for both tasks in terms of accuracy at task
dt.localization$bins.actual <- cut(dt.localization$actualMm-76,breaks = 12, labels=convertToMm(1:12))
dt.localization$bins.reported <- cut(dt.localization$responseMm-76,breaks = 12,labels=convertToMm(1:12))
dt.localization$bins.error <- cut(dt.localization$errorMm,breaks = 12)
dt.localization$bins.abserror <- cut(abs(dt.localization$errorMm),breaks = seq(0,160,20), seq(10,150,20))

dt.binnedlocalization <- dt.localization[, list(avgError = mean(errorMm), 
                                                confidence.M = mean(confidence.num),
                                                N = .N), 
                                         by = .(participant, area, bins.actual, task)]
dt.binnedlocalization.2 <- dt.localization[, list(avgError = mean(errorMm), 
                                                  confidence.M = mean(confidence.num),
                                                  N = .N), 
                                           by = .(participant, area, bins.actual, bins.reported, task)]

## modeling experiment priors/likelihoods?
# formulas from chapter 4 - the response distribution

# response distribution - distribution of the stimulus estimate when the true stimulus is s
# this might be the average response for position of the hand for each location bin
responseDist <- ggplot(data=dt.localization, aes(x=responseMm, alpha=0.2))
responseDist <- responseDist + geom_histogram(binwidth=50, fill="#E69A8DFF") + xlim(200,700)
responseDist <- responseDist + facet_wrap("participant")

stimulusDist <- ggplot(data=dt.localization, aes(x=actualMm, alpha=0.2))
stimulusDist <- stimulusDist + geom_histogram(binwidth=50, fill="#5F4B8BFF") + xlim(200,700)
stimulusDist <- stimulusDist + facet_wrap("participant")

responseDist + stimulusDist

long_data <- data.table(dt.localization %>%
  pivot_longer(cols = c(actualMm, responseMm), names_to = "type", values_to = "position"))

# mean of the stimuus distribution by area
# left_mean <- mean(long_data[type=="actualMm" & area=="Left_Shoulder"]$position)
# mid_mean <- mean(long_data[type=="actualMm" & area=="Midline"]$position)
# right_mean <- mean(long_data[type=="actualMm" & area=="Right_Shoulder"]$position)

# mean of the response distribution by area
long_data[, mean_val := mean(position), by=c("type","area","participant")]
# left_mean_r <- mean(long_data[type=="responseMm" & area=="Left_Shoulder"]$position)
# mid_mean_r <- mean(long_data[type=="responseMm" & area=="Midline"]$position)
# right_mean_r <- mean(long_data[type=="responseMm" & area=="Right_Shoulder"]$position)

# compare the different area's stimulus distributions across participants
combined_dists <- ggplot(data=long_data[type=="actualMm"], aes(x = position, fill = handed_area)) +
  geom_density(alpha = 0.5, position = "identity") + xlim(250,750) +
  scale_fill_manual(values = c("grey80","grey5","grey40"), labels = c("Body Midline", "Dominant", "Non-dominant")) +
  facet_wrap("participant", scales = "free_x") + theme_minimal(18) +
  xlab("Actual Hand Position (mm)") + ylab("Density") + labs(fill="Handed Region") +
  geom_vline(data=long_data[type=="actualMm"], aes(xintercept = mean_val), linetype=2) 
  
combined_dists

ggsave(combined_dists,
       path="~/Desktop/Bayes",
       filename = "actual_by_handed_area.pdf",
       bg = "transparent",
       width=500,
       height=500,
       units="mm")

# compare the different area's response distributions across participants
combined_dists <- ggplot(data=long_data[type=="responseMm"], aes(x = position, fill = handed_area)) +
  geom_density(alpha = 0.5, position = "identity") + xlim(250,750) +
  facet_wrap("participant", scales = "free_x") + theme_minimal(18) +
  scale_fill_manual(values = c("#5F4B8BFF","#E69A8DFF","tan"), labels=c("Body Midline", "Dominant", "Non-dominant")) +
  geom_vline(data=long_data[type=="responseMm"], aes(xintercept = mean_val, color=handed_area), linetype=2, show.legend = FALSE) +
  xlab("Reported Hand Position (mm)") + ylab("Density") + labs(fill="Handed Region") +
  scale_color_manual(values = c("dominant" = "#5F4B8BFF", "center" = "#E69A8DFF", "nondominant"="tan")) 
  
combined_dists

ggsave(combined_dists,
       path="~/Desktop/Bayes",
       filename = "response_by_handed_area.pdf",
       bg = "transparent",
       width=500,
       height=500,
       units="mm")

# compare the actual to response positions across handed areas
combined_dists <- ggplot(data=long_data, aes(x = position, fill = type)) +
  geom_density(alpha = 0.5, position = "identity") + xlim(250,750) +
  scale_color_manual(values = c("actualMm" = "#5F4B8BFF", "responseMm" = "#E69A8DFF"), labels = c("Actual", "Response")) +
  scale_fill_manual(values = c("actualMm" = "#5F4B8BFF", "responseMm" = "#E69A8DFF"),labels = c("Stimulus Distribution", "Response Distribution")) +
  facet_wrap("participant", scales = "free_x") + theme_minimal(18) + labs(fill=element_blank()) +
  geom_vline(data=long_data[type=="actualMm"], aes(xintercept = mean_val, color=type), linetype=4, show.legend=FALSE) +
  geom_vline(data=long_data[type=="responseMm"], aes(xintercept = mean_val, color=type), linetype=4, show.legend=FALSE) +
  xlab("Hand Position (mm)") + ylab("Density") 

combined_dists

ggsave(combined_dists,
       path="~/Desktop/Bayes",
       filename = "actual_vs_response_w_means.pdf",
       bg = "transparent",
       width=500,
       height=500,
       units="mm")

# two questions to answer through bayesian model comparison

# want to see whether a likelihood model with the same std for each area fits just as well as one that considers all the same 

# see if there is a prior centered at the dominant hand for responses


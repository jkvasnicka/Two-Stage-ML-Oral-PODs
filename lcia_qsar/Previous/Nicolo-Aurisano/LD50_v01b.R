library(dplyr)
library(grid)
library(gridExtra)
library(hablar)
library(readxl)
library(writexl)
library(ggplot2)
library(car)
library(rsq)
library(MASS) 
library(ggpubr)
library(boot)
library(DescTools)
library(scales)
library(magrittr)
library(reshape2)
library(cowplot)
#library(fitdistrplus) 
#library(survival) 
#library(qualityTools)
 
#Import full table directly from excel (need "readxl" library)
Human_toxval_pod_summary_with_references <- read_excel("C:/Users/niau/Desktop/Tox_data/toxval_all_with_references_2021-03-22_HUMAN.xlsx")

#List unique values in a vector
#a <- data.frame(unique(b$critical_effect))
write_xlsx(df_db_LD50,"C:/Users/niau/Desktop/file name3.xlsx")

#CURATION
{
  #### Curation (grouping) ####
  #Import Grouping tables
  {grouping_toxval_type <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/toxval_type.xlsx")
  grouping_risk_assessment_class <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/risk_assessment_class.xlsx")
  grouping_exposure_route <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/exposure_route.xlsx")
  grouping_species_original <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/species_original.xlsx")
  grouping_study_type <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/study_type.xlsx")
  grouping_CF_interspecies <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/CF_interspecies.xlsx")
  grouping_study_duration_units <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/study_duration_units.xlsx")
  grouping_exposure_route_missing <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/exposure_route_missing.xlsx")
  grouping_standardized_effect_categories <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/standardized_effect_categories.xlsx")
  grouping_conceptual_model_nrd <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/conceptual_model_nrd.xlsx")
  grouping_conceptual_model_rd <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/conceptual_model_rd.xlsx")
  grouping_U_duration <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/U_duration.xlsx")
  grouping_U_interspecies <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/U_interspecies.xlsx")
  grouping_U_toxval_type_nrd <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/U_toxval_type_nrd.xlsx")
  grouping_U_toxval_type_rd <- read_excel("C:/Users/niau/Desktop/Tox_data/grouping files/U_toxval_type_rd.xlsx")
  }
  
  #Add now column with curated entry - 
  #left_joint function is the same as match-index of Excel. first define table that you want to edit, then define table for the matching
  #NOTE columns need to have same name to work like this
  {Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_toxval_type) 
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_risk_assessment_class)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_exposure_route)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_species_original)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_study_type)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_CF_interspecies)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_study_duration_units)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_exposure_route_missing)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_standardized_effect_categories)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_conceptual_model_nrd)
    Human_toxval_pod_summary_with_references <- left_join(Human_toxval_pod_summary_with_references, grouping_conceptual_model_rd)}
  
  #### end ####
  
  #### DROP entries ####
  # ALL NOT MAMMALS
  Human_toxval_pod_summary_with_references <-  filter(Human_toxval_pod_summary_with_references, mammals_check != 0)
  
  # ALL Unique_toxval_type = DISREGARD
  Human_toxval_pod_summary_with_references <- filter(Human_toxval_pod_summary_with_references, Unique_toxval_type != "disregard") 
  
  # ALL Standardized_effect_categories = DISREGARD
  Human_toxval_pod_summary_with_references <-  filter(Human_toxval_pod_summary_with_references, Standardized_effect_categories != "disregard") 
  
  #Drop data without CAS
  Human_toxval_pod_summary_with_references <- filter(Human_toxval_pod_summary_with_references, 
                                                     casrn != "--" &
                                                       casrn != "MULTI-PL-E" &
                                                       casrn != "unkn-ow-n"&
                                                       casrn != "Vari-ou-s"&
                                                       casrn != "NOCAS_Unknown"&
                                                       casrn != "NOCAS_-"&
                                                       casrn != "NOCAS_[Name not available]" &
                                                       casrn != "NOCAS_Not applicable - multicomponent substan"&
                                                       casrn != "NOCAS_Not Applicable"&
                                                       casrn != "NOCAS_not applicable, UVCB substance"&
                                                       casrn != "NOCAS_Not applicable-UVCB" &
                                                       casrn != "NOCAS_UVCB substance not applicable")
  
  #### end ####
  
  #### Convert entries in mg/kg diet ####
  
  #all entries (toxval_value) defined in original unit as "mg/kg diet" were wrongly reported in the original file, need to apply convert them (source is ECHA/REACH)
  #Adjust entries in "mg/kg diet" by applying factor for rats
  
  adj_factor_rat <- 16 #factor calculated based on available studies
  
  Human_toxval_pod_summary_with_references$toxval_numeric_adj <- ifelse(
    Human_toxval_pod_summary_with_references$toxval_units_original == "mg/kg diet" 
    & (Human_toxval_pod_summary_with_references$Unique_species_original== "rat" | Human_toxval_pod_summary_with_references$Unique_species_original== "rat*"),
    Human_toxval_pod_summary_with_references$toxval_numeric / adj_factor_rat,
    Human_toxval_pod_summary_with_references$toxval_numeric)
  
  #Adjust entries in "mg/kg diet" by applying factor for mouse
  adj_factor_mouse <- 4.5  #factor calculated based on available studies
  Human_toxval_pod_summary_with_references$toxval_numeric_adj <- ifelse(
    Human_toxval_pod_summary_with_references$toxval_units_original == "mg/kg diet" 
    & Human_toxval_pod_summary_with_references$Unique_species_original== "mouse",
    Human_toxval_pod_summary_with_references$toxval_numeric_adj / adj_factor_mouse,
    Human_toxval_pod_summary_with_references$toxval_numeric_adj)
  #### end ####
  
  #### Extrapolation to human ####
  #we need to extrapolate all reported values to human
  #we do that by dividing by the pre-defined factors
  #NOTE some values already reported for humans fro this reason we exclude them from extrapolation with ifelse function
  Human_toxval_pod_summary_with_references$toxval_numeric_extrap_H <- ifelse(
    Human_toxval_pod_summary_with_references$Unique_toxval_type == "NOAEL(h)*" |
      Human_toxval_pod_summary_with_references$Unique_toxval_type == "BMCL(h)*" |  
      Human_toxval_pod_summary_with_references$Unique_toxval_type == "BMDL(h)*" ,
    Human_toxval_pod_summary_with_references$toxval_numeric_adj,
    Human_toxval_pod_summary_with_references$toxval_numeric_adj / Human_toxval_pod_summary_with_references$CF_interspecies)
  #### end ####
  
  #### Convert exposure duration ####
  Human_toxval_pod_summary_with_references$study_duration_value_day <- ifelse(
    Human_toxval_pod_summary_with_references$study_duration_units == "day" |
      Human_toxval_pod_summary_with_references$study_duration_units == "year" |
      Human_toxval_pod_summary_with_references$study_duration_units == "week" |
      Human_toxval_pod_summary_with_references$study_duration_units == "month" |
      Human_toxval_pod_summary_with_references$study_duration_units == "hour" |
      Human_toxval_pod_summary_with_references$study_duration_units == "minute",
    Human_toxval_pod_summary_with_references$study_duration_value / as.numeric(Human_toxval_pod_summary_with_references$CF_duration),
    Human_toxval_pod_summary_with_references$study_duration_value)
  
  #### end ####
  
  #### Fix exposure route missing ####
  
  Human_toxval_pod_summary_with_references$exposure_route_specfied_missing <- ifelse(
    Human_toxval_pod_summary_with_references$Unique_exposure_route == "-",
    Human_toxval_pod_summary_with_references$exposure_route_specified,
    Human_toxval_pod_summary_with_references$Unique_exposure_route)
  
  #and again
  
  Human_toxval_pod_summary_with_references$Unique_risk_assessment_class <- ifelse(
    Human_toxval_pod_summary_with_references$Unique_risk_assessment_class == "repeat dose",
    Human_toxval_pod_summary_with_references$Unique_risk_assessment_class_RD,
    Human_toxval_pod_summary_with_references$Unique_risk_assessment_class
  )
  #### end ####
  
  #drop data with weird units
  {Human_toxval_pod_summary_with_references <- filter(Human_toxval_pod_summary_with_references, 
                                                      toxval_units != "-"        &
                                                        toxval_units != "other:"        &
                                                        toxval_units != "uM"        &
                                                        toxval_units != "M"        &
                                                        toxval_units != "Other" &
                                                        toxval_units != "%" &
                                                        toxval_units !=  "% v/v" &
                                                        toxval_units !=  "NR" & 
                                                        toxval_units !=  "No Data" &
                                                        toxval_units !=   "percentage" &
                                                        toxval_units !=  "no units"    &
                                                        toxval_units !=  "percentage(diet)" &
                                                        toxval_units !=  "% of diet" &
                                                        toxval_units !=  "% w/v"   &
                                                        toxval_units !=  "% vol"   &
                                                        toxval_units !=  "% w/w" &
                                                        toxval_units !=  "ppm brain BG" &
                                                        toxval_units !=  "ppm brain CB"&
                                                        toxval_units !=  "ppm brain HC"&
                                                        toxval_units !=  "ppm brain HT"&
                                                        toxval_units !=  "ppm brain MB"&
                                                        toxval_units !=  "ppm brain MO"&
                                                        toxval_units !=  "ppm brain cortex"&
                                                        toxval_units !=  "ppm plasma"    &
                                                        toxval_units !=   "ppm serum, day 15" &
                                                        toxval_units !=  "ppm serum, day 30" &
                                                        toxval_units !=  "mg/L serum" &
                                                        toxval_units !=  "ug/g brain CB" &
                                                        toxval_units !=  "ug/g brain CC" &
                                                        toxval_units !=  "ug/g brain HC" &
                                                        toxval_units !=  "ug/g brain MO" &
                                                        toxval_units !=  "ug/g bone" &
                                                        toxval_units !=  "unitless" &
                                                        toxval_units !=  "2" &
                                                        toxval_units !=  "26" &
                                                        toxval_units != "12" &
                                                        toxval_units != "32" &
                                                        toxval_units != "18" &
                                                        toxval_units != "9" &
                                                        toxval_units != "84" &
                                                        toxval_units != "pCi/g SEDIMENT" &
                                                        toxval_units != "pCi/g SOIL" &
                                                        toxval_units != "pCi/L WATER" &
                                                        toxval_units != "mSv" &
                                                        toxval_units != "mSv/yr" &
                                                        toxval_units != "fibers/cc"&
                                                        toxval_units != "percentage(drinking water)"&
                                                        toxval_units != "percentage(diet by weight)"&
                                                        toxval_units != "percentage(weight)"&
                                                        toxval_units != "<c4>?<c2>?<c3>?<c2>?<c4>?<c2>?<c3>?XXX?XXX?XX"&
                                                        toxval_units != "g/37.9L/0.1 ha"&
                                                        toxval_units != "units" &
                                                        toxval_units != "N" &
                                                        toxval_units != "oz/25lbs bdwt"&
                                                        toxval_units != "mg%"&
                                                        toxval_units != "%(CrO3)"&
                                                        toxval_units != "fibers/lwater^-1" ) 
  }
  
  
  #### Fix effect REP/DEV ####
  
  Human_toxval_pod_summary_with_references$Unique_study_type <- ifelse(
    Human_toxval_pod_summary_with_references$Standardized_effect_categories == "development"|
      Human_toxval_pod_summary_with_references$Standardized_effect_categories == "reproduction",
    "reproductive developmental",
    Human_toxval_pod_summary_with_references$Unique_study_type)
  
  #### end ####
  
  #### Convert NOAEL with qualifier "<" to LOAEL ####
  
  Human_toxval_pod_summary_with_references$Unique_toxval_type <- ifelse(
    Human_toxval_pod_summary_with_references$Unique_toxval_type == "NOAEL"   &
      Human_toxval_pod_summary_with_references$toxval_numeric_qualifier == "<",
    "LOAEL",
    Human_toxval_pod_summary_with_references$Unique_toxval_type)
  
  #### end ####
  
  #### DROP useless variables from curated dataset ####
  # with the function subset you first define the starting table, then using "select = -c()" you indicates the column that you do NOT want
  
  Human_toxval_pod_summary_with_references = 
    subset(Human_toxval_pod_summary_with_references, 
           select = -c(toxval_numeric_original, exposure_route_specified,toxval_id,mammals_check,  CF_duration))
  #### end ####
  
}

#### CREATE SUB-SET ####
#sub-set
{db_LD50 <- subset(Human_toxval_pod_summary_with_references,
                               Unique_toxval_type == "LD50" 
                             &  exposure_route_specfied_missing == "oral")

#Drop columns related to conceptual models for rep and dev
db_LD50 = subset(db_LD50, 
          select = -c(
           Conceptual_model_rd_1, 
           Conceptual_model_rd_2,
           Conceptual_model_nrd_1, 
           Conceptual_model_nrd_2))}

#curate subset wrong units and empty values
{db_LD50 <- filter(db_LD50, 
                               toxval_units_original == "mg/kg bw"              |
                               toxval_units_original == "mg/kg"                 |
                               toxval_units_original == "mg/kg-bw"              |
                               toxval_units_original == "mg/kg bw/day"          |
                               toxval_units_original == "mg/kg-day (nominal)"   |
                               toxval_units_original == "mg/kg-day"             |
                               toxval_units_original == "mg/kg bdwt") 
  
db_LD50 <- filter(db_LD50, toxval_numeric_extrap_H != "") }

db_LD50$log_toxval_numeric_extrap_H <- log10(db_LD50$toxval_numeric_extrap_H)

df_db_LD50 <- db_LD50  %>%
  group_by(casrn) %>%
  summarise(count_LD50 = sum(Unique_toxval_type == "LD50"),
            median_LD50 = median(log_toxval_numeric_extrap_H),
            goem_LD50 = mean(log_toxval_numeric_extrap_H),
            SDV_LD50 = sd(log_toxval_numeric_extrap_H))

SDV_fixed <- 0.3
df_db_LD50$LD50_25 <-ifelse(df_db_LD50$count_LD50 < 11,
                           qnorm(0.25, df_db_LD50$median_LD50, SDV_fixed),
                           qnorm(0.25, df_db_LD50$median_LD50, df_db_LD50$SDV_LD50))

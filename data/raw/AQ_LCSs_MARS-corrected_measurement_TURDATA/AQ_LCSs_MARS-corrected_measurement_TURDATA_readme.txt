Air quality low-cost sensors (AQ LCSs) - MARS corrected 1hour data:
- dt_beg_utc and dt_end_utc are timestamps for the beginning and the end of the measurement interval; both using UTC time; timestamp format dd.mm.yyyy hh:mm
- LCS concentrations corrected by Multivariate Adaptive Regression Splines (MARS) method for each LCS named as: pollutant_S2C, pollutant_S3C,…, pollutant_S20C
- measurement location: RM_Prague_4-Libus and Legerova_domain
- measurement_program: Initial_comparative_measurement, Legerova_campaign and Final_comparative_measurement
- during Inital_comparative_measurement (16.12.2021 07:00 - 30.05.2022 05:00) all LCSs units were collocated at RM Prague 4-Libus
- during Legerova_campaign (30.05.2022 06:00 - 28.03.2023 06:00) LCSs were placed at various localities and height levels AGL within the Legerova domain
- during Final_comparative_measurement (09.05.2023 11:00 - 14.06.2023 08:00) all LCSs units were again collocated at RM Prague 4-Libus
- for the complete list of LCSs placement and their measurement periods during Legerova_campaign see the file TURDATA_metadata.xlsx

Notes:
- NO2 and O3 concentration units are in ppb
- PM10 and PM2.5 concentration units are in µg·m−3
- missing values are marked as NA
- be aware of the data gap between the end of Legerova_campaign and start of Final_comparative measurement (not marked as NA)
- The LCSs S1, S8 and S17 are missing in all the datasets (broken LCS units)
- The LCS S6 was collocated during the whole measurement period at the Prague 4-Libus RM 
- The LCS S4 was since 24 March 2022 collocated at the Prague 2-Legerova RM 
- In case of LCS S15 the data from final comparative measurement at the Prague 4-Libus RM are missing (sensor failure)
- In NO2_S9C a significant data drift was detected during the measurement campaign, these data should be considered as invalid during Legerova campaign and final comparative measurement

A) MARS correction calculation:
- MARS correction equations were calculated for each LCS individually based on the data gained during the whole Initial_comparative_measurement 
at Prague 4-Libus RM 
- the program Statistica v13.1.0 (TIBCO) was used for calculation; alternatively the earth package in R software (Milborrow, 2023) can be used 
- calculation of correction equations:
1) dependent varible: NO2_RM; independent variables: NO2_SxR + TMP + RH + WV + GLRD + hour of the day 
2) dependent varible: O3_RM; independent variables: O3_SxR + O3_SxR/NO2_SxR + TMP + RH + WV + GLRD + hour of the day 
3) dependent varible: PM10_EM; independent variables: PM10_SxR + TMP + RH + WV + GLRD + hour of the day 
4) dependent varible: PM2_5_EM; independent variables: PM2_5_SxR + TMP + RH + WV + GLRD + hour of the day

explanations:
pollutant_RM = concentrations measured by reference monitor during Inital_comparative_measurement at Prague 4-Libus RM
pollutant_EM = concentrations measured by equivalent monitor during Inital_comparative_measurement at Prague 4-Libus RM
pollutant_SxR = raw measured LCS concentrations of particular sensor during Inital_comparative_measurement at Prague 4-Libus RM
O3_SxR/NO2_SxR = ratio of raw measured O3 and NO2 LCS concentrations of particular sensor during Inital_comparative_measurement at Prague 4-Libus RM
TMP = 1hour temperature measured at Prague Libus MS
RH = 1hour relative humidity measured at Prague Libus MS
WV = 1hour wind velocity measured at Prague Libus MS
GLRD = 1hour global radiance intensity measured at Prague Libus MS

B) MARS corrections application to Legerova campaign
- MARS correction equations calculated during the Initial_comparative_measumerement were applied to the raw measured LCS concentrations of particualr sensors 
during Legerova campaign including meteorological data measured at Prague Karlov MS
- data from RM and EM Legerova were not used for the correction calculation itself, but only for the indicative comparison with raw and corrected LCS measurements 
gained during Legerova_campaign
- because there is no O3_RM measurement available at RM_Prague_2-Legerova, the raw and corrected O3 LCS measurements gained during Legerova_campaign were 
indicatively compared to the O3 measurement at RM_Prague_9-Vysocany
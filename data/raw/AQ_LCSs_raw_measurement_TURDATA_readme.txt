Air quality low-cost sensors (AQ LCSs) - raw 1hour data:
- dt_beg_utc and dt_end_utc are timestamps for the beginning and the end of the measurement interval; both using UTC time; timestamp format dd.mm.yyyy hh:mm
- raw concentrations measured by each LCS named as: pollutant_S2R, pollutant_S3R,…, pollutant_S20R
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
- be aware of the data gap between the end of Legerova_camppaign and start of Final_comparative measurement (not marked as NA)
- The LCSs S1, S8 and S17 are missing in all the datasets (broken LCS units)
- The LCS S6 was collocated during the whole measurement period at the Prague 4-Libus RM 
- The LCS S4 was since 24 March 2022 collocated at the Prague 2-Legerova RM 
- In case of LCS S15 the data from final comparative measurement at the Prague 4-Libus RM are missing (sensor failure)
- In NO2_S9R a significant data drift was detected during the measurement campaign, these data should be considered as invalid during Legerova campaign and final comparative measurement 

 
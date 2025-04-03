#%% 2018 -2020
import pandas as pd

df_2018_gsec10_1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\HH\GSEC10_1.csv")
df_2019_gsec10_1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\HH\gsec10_1.csv")
#%% adding year and missing columns
if 's10q04' not in df_2019_gsec10_1.columns:
    df_2019_gsec10_1['s10q04'] = pd.NA

df_2018_gsec10_1['year'] = 2018
df_2019_gsec10_1['year'] = 2019

df_gsec10_1 = pd.concat([df_2018_gsec10_1, df_2019_gsec10_1], ignore_index=True)

#%% 
df_2018_gsec10_3 = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\HH\GSEC10_3.csv")
df_2019_gsec10_3 = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\HH\gsec10_3.csv")

if 't0_hhid' in df_2018_gsec10_3.columns:
    df_2018_gsec10_3 = df_2018_gsec10_3.drop(columns=['t0_hhid'])

cols_gsec10_3 = ['hhid', 's10q13', 's10q14', 's10q15a', 's10q15b', 
                 's10q15c', 's10q16', 's10q17a', 's10q17b', 's10q17c']

df_2018_gsec10_3 = df_2018_gsec10_3[cols_gsec10_3]
df_2019_gsec10_3 = df_2019_gsec10_3[cols_gsec10_3]

df_2018_gsec10_3['year'] = 2018
df_2019_gsec10_3['year'] = 2019


df_gsec10_3 = pd.concat([df_2018_gsec10_3, df_2019_gsec10_3], ignore_index=True)

# %% 2015, renaming and dropping extra columns

df_2015_gsec10_1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\household\gsec10_1.csv")

#rename
rename_dict_gsec10_1 = {
    "h10q1": "s10q01",
    "h10q2": "s10q02",
    "h10q3": "s10q03",
    "h10q4": "s10q04",
    "h10q5a": "s10q05a",
    "h10q5b": "s10q05b",
    "h10q6": "s10q06",
    "h10q7a": "s10q07a",
    "h10q7b": "s10q07b",
    "h10q9": "s10q09",
    "h10q10": "s10q10",
    "h10q11": "s10q11",
    "h10q12": "s10q12"
}

df_2015_gsec10_1.rename(columns=rename_dict_gsec10_1, inplace=True)

if 's10q05c' not in df_2015_gsec10_1.columns:
    df_2015_gsec10_1['s10q05c'] = pd.NA
for extra in ['h10q7c', 'h10q7d', 'hh']:
    if extra in df_2015_gsec10_1.columns:
        df_2015_gsec10_1.drop(columns=extra, inplace=True)

#adding year 
df_2015_gsec10_1['year'] = 2015

#%% 
df_2015_gsec10_3 = pd.read_csv(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\household\gsec10_3.csv")
if 'HHID' in df_2015_gsec10_3.columns:
    df_2015_gsec10_3.rename(columns={'HHID': 'hhid'}, inplace=True)
if 'HOUSEHOLD CODE' in df_2015_gsec10_3.columns:
    df_2015_gsec10_3.drop(columns=['HOUSEHOLD CODE'], inplace=True)

rename_dict_gsec10_3 = {
    "h10q13": "s10q13",
    "h10q14": "s10q14",
    "h10q15a": "s10q15a",
    "h10q15b": "s10q15b",
    "h10q15c": "s10q15c",
    "h10q16": "s10q16",
    "h10q17a": "s10q17a",
    "h10q17b": "s10q17b",
    "h10q17c": "s10q17c"
}

df_2015_gsec10_3.rename(columns=rename_dict_gsec10_3, inplace=True)

cols_gsec10_3 = ['hhid', 's10q13', 's10q14', 's10q15a', 's10q15b', 
                 's10q15c', 's10q16', 's10q17a', 's10q17b', 's10q17c']
df_2015_gsec10_3 = df_2015_gsec10_3[cols_gsec10_3]
df_2015_gsec10_3['year'] = 2015

#%% combining all years

df_gsec10_1_all = pd.concat([df_2015_gsec10_1, df_2018_gsec10_1, df_2019_gsec10_1], ignore_index=True)
df_gsec10_3_all = pd.concat([df_2015_gsec10_3, df_2018_gsec10_3, df_2019_gsec10_3], ignore_index=True)

# write to csv
# df_gsec10_1_all.to_csv(r"D:\GWU_4th_Sem\code\output_gsec10_1_all_years.csv", index=False)
# df_gsec10_3_all.to_csv(r"D:\GWU_4th_Sem\code\output_gsec10_3_all_years.csv", index=False)

#%% 2013


df_2013_gsec10_1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\GSEC10_1.csv")

# Rename household
if 'HHID' in df_2013_gsec10_1.columns:
    df_2013_gsec10_1.rename(columns={'HHID': 'hhid'}, inplace=True)
if 'Household number' in df_2013_gsec10_1.columns:
    df_2013_gsec10_1.drop(columns=['Household number'], inplace=True)

# Rename variables 
rename_dict_gsec10_1_2013 = {
    "h10q1": "s10q01",
    "h10q2": "s10q02",
    "h10q3": "s10q03",
    "h10q4": "s10q04",
    "h10q5a": "s10q05a",
    "h10q5b": "s10q05b",
    "h10q6": "s10q06",
    "h10q7a": "s10q07a",
    "h10q7b": "s10q07b",
    "h10q9": "s10q09",
    "h10q10": "s10q10",
    "h10q11": "s10q11",
    "h10q12": "s10q12"
}
df_2013_gsec10_1.rename(columns=rename_dict_gsec10_1_2013, inplace=True)
if 's10q05c' not in df_2013_gsec10_1.columns:
    df_2013_gsec10_1['s10q05c'] = pd.NA
for extra in ['h10q7c', 'h10q7d', 'wgt_X']:
    if extra in df_2013_gsec10_1.columns:
        df_2013_gsec10_1.drop(columns=extra, inplace=True)

# adding year
df_2013_gsec10_1['year'] = 2013

df_2013_gsec10_3 = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\GSEC10_3.csv")

# Renaming household ID 
if 'HHID' in df_2013_gsec10_3.columns:
    df_2013_gsec10_3.rename(columns={'HHID': 'hhid'}, inplace=True)
if 'HOUSEHOLD CODE' in df_2013_gsec10_3.columns:
    df_2013_gsec10_3.drop(columns=['HOUSEHOLD CODE'], inplace=True)

rename_dict_gsec10_3_2013 = {
    "h10q13": "s10q13",
    "h10q14": "s10q14",
    "h10q15a": "s10q15a",
    "h10q15b": "s10q15b",
    "h10q15c": "s10q15c",
    "h10q16": "s10q16",
    "h10q17a": "s10q17a",
    "h10q17b": "s10q17b",
    "h10q17c": "s10q17c"
}
df_2013_gsec10_3.rename(columns=rename_dict_gsec10_3_2013, inplace=True)
cols_gsec10_3 = ['hhid', 's10q13', 's10q14', 's10q15a', 's10q15b', 
                 's10q15c', 's10q16', 's10q17a', 's10q17b', 's10q17c']
df_2013_gsec10_3 = df_2013_gsec10_3[cols_gsec10_3]

df_2013_gsec10_3['year'] = 2013

# -------------------- Combine 2013 with 2015, 2018, & 2019 --------------------
df_gsec10_1_final = pd.concat([df_2013_gsec10_1, df_gsec10_1], ignore_index=True)
df_gsec10_3_final = pd.concat([df_2013_gsec10_3, df_gsec10_3], ignore_index=True)

# -------------------- Save the Final Combined Files --------------------
df_gsec10_1_final.to_csv(r"..\Datasets\Uganda\output_gsec10_1_all_years.csv", index=False)
df_gsec10_3_final.to_csv(r"..\Datasets\Uganda\output_gsec10_3_all_years.csv", index=False)
# %%

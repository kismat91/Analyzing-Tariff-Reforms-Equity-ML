#%% 2018 -2020
import pandas as pd
# 2018
df_2018_gsec10_1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\HH\GSEC10_1.csv")
df_2018_gsec1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\HH\GSEC1.csv")
df_2018_gsec4 = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\HH\GSEC4.csv") 
df_2018_gsec7_2 = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\HH\GSEC7_2.csv")
df_2018_gsec14 = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\HH\GSEC14.csv")
df_2018_pov = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\pov2018_19.csv")
df_2018_gsec10_3 = pd.read_csv(r"..\Datasets\Uganda\UGA_2018_UNPS_v02_M_CSV\HH\GSEC10_3.csv")
# 2019
df_2019_gsec10_1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\HH\gsec10_1.csv")
df_2019_gsec1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\HH\GSEC1.csv")
df_2019_gsec4 = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\HH\GSEC4.csv")
df_2019_gsec7_2 = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\HH\GSEC7_2.csv")
df_2019_gsec14 = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\HH\GSEC14.csv")
df_2019_pov = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\pov2019_20.csv")
df_2019_gsec10_3 = pd.read_csv(r"..\Datasets\Uganda\UGA_2019_UNPS_v03_M_CSV\HH\gsec10_3.csv")
#2015
df_2015_gsec10_3 = pd.read_csv(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\household\gsec10_3.csv")
df_2015_gsec10_1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\household\gsec10_1.csv")
df_2015_gsec1 = pd.read_stata(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\household\gsec1.dta")
df_2015_gsec4 = pd.read_stata(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\household\gsec4.dta")
df_2015_gsec11_2 = pd.read_stata(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\household\gsec11_2.dta")
df_2015_gsec14 = pd.read_stata(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\household\gsec14.dta")
df_2015_pov = pd.read_stata(r"..\Datasets\Uganda\UGA_2015_UNPS_v02_M_STATA14\pov2015_16.dta") 
#2013
df_2013_gsec10_1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\GSEC10_1.csv")
df_2013_gsec10_3 = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\GSEC10_3.csv")
df_2013_gsec1 = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\GSEC1.csv")
df_2013_gsec4 = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\GSEC4.csv",encoding='latin1')
df_2013_gsec11_A = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\GSEC11A.csv")
df_2013_gsec14A = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\GSEC14A.csv")
df_2013_pov = pd.read_csv(r"..\Datasets\Uganda\UGA_2013_UNPS_v02_M_CSV\pov2013_14.csv")

###############################################################################################################
#Economic data 
###############################################################################################################
#%% merging economic data
# merged_2018 = df_2018_gsec1.merge(df_2018_gsec4, on="hhid", how="outer")
# merged_2018 = merged_2018.merge(df_2018_gsec7_2, on="hhid", how="outer")
# merged_2018 = merged_2018.merge(df_2018_gsec14, on="hhid", how="outer")
# df_2018_eco = merged_2018.merge(df_2018_pov, on="hhid", how="outer")

# #%% merging economic data for 2019
# merged_2019 = df_2019_gsec1.merge(df_2019_gsec4, on="hhid", how="outer")
# merged_2019 = merged_2019.merge(df_2019_gsec7_2, on="hhid", how="outer")
# merged_2019 = merged_2019.merge(df_2019_gsec14, on="hhid", how="outer")
# df_2019_eco = merged_2019.merge(df_2019_pov, on="hhid", how="outer")

# #%% merging economic data for 2015


# df_2015_gsec4 = df_2015_gsec4.drop(columns=['hh'], errors='ignore')
# df_2015_gsec11_2 = df_2015_gsec11_2.drop(columns=['hh'], errors='ignore')
# df_2015_gsec14 = df_2015_gsec14.drop(columns=['hh'], errors='ignore')


# merged_2015 = df_2015_gsec1.merge(df_2015_gsec4, on="hhid", how="outer")
# merged_2015 = merged_2015.merge(df_2015_gsec11_2, on="hhid", how="outer")
# merged_2015 = merged_2015.merge(df_2015_gsec14, on="hhid", how="outer")
# df_2015_eco = merged_2015.merge(df_2015_pov, on="hh", how="outer")

# #%% merging economic data for 2013
# df_2013_gsec1.rename(columns={'HHID': 'hhid'}, inplace=True)
# df_2013_gsec4.rename(columns={'HHID': 'hhid'}, inplace=True)
# df_2013_gsec11_A.rename(columns={'HHID': 'hhid'}, inplace=True)
# df_2013_gsec14A.rename(columns={'HHID': 'hhid'}, inplace=True)
# df_2013_pov.rename(columns={'HHID': 'hhid'}, inplace=True)
# df_2013_gsec4.rename(columns={'wgt_X': 'wgt_X_gsec4'}, inplace=True)
# df_2013_gsec11_A.rename(columns={'wgt_X': 'wgt_X_gsec11_A'}, inplace=True)
# df_2013_gsec14A.rename(columns={'wgt_X': 'wgt_X_gsec14A'}, inplace=True)
# df_2013_pov.rename(columns={'wgt_X': 'wgt_X_pov'}, inplace=True)

# merged_2013 = df_2013_gsec1.merge(df_2013_gsec4, on="hhid", how="outer")
# merged_2013 = merged_2013.merge(df_2013_gsec11_A, on="hhid", how="outer")
# merged_2013 = merged_2013.merge(df_2013_gsec14A, on="hhid", how="outer")
# df_2013_eco = merged_2013.merge(df_2013_pov, on="hhid", how="outer")



# #%%

# #%% adding year
# df_2018_eco['year'] = 2018      
# df_2019_eco['year'] = 2019
# df_2015_eco['year'] = 2015
# df_2013_eco['year'] = 2013
# # %% cleaning and renaming
# # 2019
# df_2019 = df_2019_eco.copy()
# df_2019.rename(columns={**{
#     "s4q05": "attended_school",
#     "s4q06": "reason_no_school",
#     "s4q04": "can_read_write",
#     "s4q07": "highest_grade",
#     "s4q08": "reason_left_school",
#     "s4q17": "funding_source"
# }, **{
#     "IncomeSource": "income_source_code",
#     "s11q04": "income_received",
#     "s11q05": "income_cash",
#     "s11q06": "income_in_kind"
# }, **{
#     "h14q02": "asset_type",
#     "h14q03": "asset_owned",
#     "h14q04": "asset_quantity",
#     "h14q05": "asset_value"
# }}, inplace=True)
# df_2019["region"] = df_2019.get("region_x", df_2019.get("region_y", pd.NA))
# df_2019["regurb"] = df_2019.get("regurb_x", df_2019.get("regurb_y", pd.NA))
# df_2019["district"] = df_2019.get("district_x", df_2019.get("district_y", pd.NA))
# df_2019["urban"] = df_2019.get("urban_x", df_2019.get("urban_y", pd.NA))
# df_2019["subregion"] = df_2019.get("subreg_x", df_2019.get("subreg_y", pd.NA))

# # 2018
# df_2018 = df_2018_eco.copy()
# df_2018.rename(columns={**{
#     "s4q05": "attended_school",
#     "s4q06": "reason_no_school",
#     "s4q04": "can_read_write",
#     "s4q07": "highest_grade",
#     "s4q08": "reason_left_school",
#     "s4q17": "funding_source"
# }, **{
#     "IncomeSource": "income_source_code",
#     "s11q04": "income_received",
#     "s11q05": "income_cash",     
#     "s11q06": "income_in_kind"
# }, **{
#     "h14q02": "asset_type",
#     "h14q03": "asset_owned",
#     "h14q04": "asset_quantity",
#     "h14q05": "asset_value"
# }}, inplace=True)
# df_2018["region"] = df_2018.get("region_x", pd.NA)
# df_2018["regurb"] = df_2018.get("regurb_x", pd.NA)
# df_2018["district"] = df_2018.get("district_code", pd.NA)
# df_2018["urban"] = df_2018.get("urban", pd.NA)
# df_2018["subregion"] = df_2018.get("subreg_x", pd.NA)
# if "income_cash" not in df_2018.columns:
#     df_2018["income_cash"] = pd.NA
# if "income_in_kind" not in df_2018.columns:
#     df_2018["income_in_kind"] = pd.NA

# # 2015
# df_2015 = df_2015_eco.copy()
# df_2015.rename(columns={**{
#     "h4q5": "attended_school",
#     "h4q6": "reason_no_school",
#     "h4q4": "can_read_write",
#     "h4q7": "highest_grade",
#     "h4q8": "reason_left_school",
#     "h4q17": "funding_source"
# }, **{
#     "s11q3": "income_source_code",
#     "s11q4": "income_received",
#     "s11q5": "income_cash",
#     "s11q6": "income_in_kind"
# }, **{
#     "h14q2": "asset_type",
#     "h14q3": "asset_owned",
#     "h14q4": "asset_quantity",
#     "h14q5": "asset_value"
# }}, inplace=True)
# df_2015["region"] = df_2015.get("region_x", pd.NA)
# df_2015["regurb"] = df_2015.get("regurb_x", pd.NA)
# df_2015["district"] = df_2015.get("district_x", pd.NA)
# df_2015["urban"] = df_2015.get("urban_x", pd.NA)
# df_2015["subregion"] = df_2015.get("sregion", pd.NA)

# # 2013
# df_2013 = df_2013_eco.copy()
# df_2013.rename(columns={**{
#     "h4q5": "attended_school",
#     "h4q6": "reason_no_school",
#     "h4q4": "can_read_write",
#     "h4q7": "highest_grade",
#     "h4q8": "reason_left_school",
#     "h4q17": "funding_source"
# }, **{
#     "h11q2": "income_source_code",
#     "h11q4": "income_received",
#     "h11q5": "income_cash",
#     "h11q6": "income_in_kind"
# }, **{
#     "h14q2": "asset_type",
#     "h14q3": "asset_owned",
#     "h14q4": "asset_quantity",
#     "h14q5": "asset_value"
# }}, inplace=True)
# df_2013["region"] = df_2013.get("region_x", pd.NA)
# df_2013["regurb"] = df_2013.get("regurb_x", pd.NA)
# df_2013["district"] = df_2013.get("district_code", pd.NA)
# df_2013["urban"] = df_2013.get("urban_x", pd.NA)
# df_2013["subregion"] = df_2013.get("sregion", pd.NA)

# final_cols = [
#     # Identifiers and time
#     "hhid", "year",
#     # Education (standardized)
#     "attended_school", "reason_no_school", "can_read_write",
#     "highest_grade", "reason_left_school", "funding_source",
#     # Income (standardized)
#     "income_source_code", "income_received", "income_cash", "income_in_kind",
#     # Assets (standardized)
#     "asset_type", "asset_owned", "asset_quantity", "asset_value",
#     # Geography
#     "region", "regurb", "subregion", "district", "urban",
#     # Household characteristics from POV
#     "hsize", "equiv", "nrrexp30", "cpexp30", "welfare", "plinen", "spline"
# ]

# def ensure_columns(df, cols):
#     for col in cols:
#         if col not in df.columns:
#             df[col] = pd.NA
#     return df

# # Ensure these final columns exist in each DataFrame:
# df_2019_final = ensure_columns(df_2019.copy(), final_cols)[final_cols]
# df_2018_final = ensure_columns(df_2018.copy(), final_cols)[final_cols]
# df_2015_final = ensure_columns(df_2015.copy(), final_cols)[final_cols]
# df_2013_final = ensure_columns(df_2013.copy(), final_cols)[final_cols]

# # -----------------------
# # Vertically stack the selected key columns from all waves
# # -----------------------
# df_all_final = pd.concat([df_2019_final, df_2018_final, df_2015_final, df_2013_final], ignore_index=True)

# # Save the final dataset:
# df_all_final.to_csv(r"..\Datasets\Uganda\combined_key_variables.csv", index=False)


# %%
def agg_mode(series):
    """Return the mode (most common value) of a series; if multiple modes, take the first; if empty, return NA."""
    s = series.dropna()
    if s.empty:
        return pd.NA
    modes = s.mode()
    return modes.iloc[0] if not modes.empty else s.iloc[0]

def agg_sum(series):
    """Return the sum of a series. If all values are missing, returns NA."""
    return series.sum(min_count=1)


# Education Aggregation
def aggregate_education_18_19(df_edu):
    # 2019/2018: GSEC4 uses s4q05, s4q06, s4q04, s4q07, s4q08, s4q17
    edu_agg = df_edu.groupby("hhid").agg({
        "s4q05": agg_mode,           # Has attended school (categorical)
        "s4q06": agg_mode,           # Reason for not attending school
        "s4q04": agg_mode,           # Literacy status
        "s4q07": "max",              # Highest grade completed (take max)
        "s4q08": agg_mode,           # Reason left school
        "s4q17": agg_mode            # Main source of funding
    }).reset_index()
    edu_agg.rename(columns={
        "s4q05": "attended_school",
        "s4q06": "reason_no_school",
        "s4q04": "can_read_write",
        "s4q07": "highest_grade",
        "s4q08": "reason_left_school",
        "s4q17": "funding_source"
    }, inplace=True)
    return edu_agg

def aggregate_education_15_13(df_edu):
    # 2015/2013: Use h4q5, h4q6, h4q4, h4q7, h4q8, h4q17
    edu_agg = df_edu.groupby("hhid").agg({
        "h4q5": agg_mode,
        "h4q6": agg_mode,
        "h4q4": agg_mode,
        "h4q7": "max",
        "h4q8": agg_mode,
        "h4q17": agg_mode
    }).reset_index()
    edu_agg.rename(columns={
        "h4q5": "attended_school",
        "h4q6": "reason_no_school",
        "h4q4": "can_read_write",
        "h4q7": "highest_grade",
        "h4q8": "reason_left_school",
        "h4q17": "funding_source"
    }, inplace=True)
    return edu_agg

# Income Aggregation
def aggregate_income_18_19(df_inc):
    # Ensure that the expected columns exist. For 2018 these may be missing.
    for col in ["s11q05", "s11q06"]:
        if col not in df_inc.columns:
            df_inc[col] = pd.NA
    # 2019/2018: GSEC7_2: IncomeSource, s11q04, s11q05, s11q06
    inc_agg = df_inc.groupby("hhid").agg({
        "IncomeSource": lambda x: ";".join(x.dropna().astype(str).unique()),
        "s11q04": agg_mode,     # Aggregation for the indicator
        "s11q05": agg_sum,      # Aggregate cash income
        "s11q06": agg_sum       # Aggregate in-kind income
    }).reset_index()
    inc_agg.rename(columns={
        "IncomeSource": "income_source_code",
        "s11q04": "income_received",
        "s11q05": "income_cash",
        "s11q06": "income_in_kind"
    }, inplace=True)
    return inc_agg

def aggregate_income_15(df_inc):
    # 2015: GSEC11_2: s11q3, s11q4, s11q5, s11q6
    inc_agg = df_inc.groupby("hhid").agg({
        "s11q3": lambda x: ";".join(x.dropna().astype(str).unique()),
        "s11q4": agg_mode,
        "s11q5": agg_sum,
        "s11q6": agg_sum
    }).reset_index()
    inc_agg.rename(columns={
        "s11q3": "income_source_code",
        "s11q4": "income_received",
        "s11q5": "income_cash",
        "s11q6": "income_in_kind"
    }, inplace=True)
    return inc_agg

def aggregate_income_13(df_inc):
    # 2013: GSEC11_A: h11q2, h11q4, h11q5, h11q6
    inc_agg = df_inc.groupby("hhid").agg({
        "h11q2": lambda x: ";".join(x.dropna().astype(str).unique()),
        "h11q4": agg_mode,
        "h11q5": agg_sum,
        "h11q6": agg_sum
    }).reset_index()
    inc_agg.rename(columns={
        "h11q2": "income_source_code",
        "h11q4": "income_received",
        "h11q5": "income_cash",
        "h11q6": "income_in_kind"
    }, inplace=True)
    return inc_agg

# Assets Aggregation
def aggregate_assets_18_19(df_asset):
    # 2019/2018: GSEC14: h14q02, h14q03, h14q04, h14q05
    asset_agg = df_asset.groupby("hhid").agg({
        "h14q02": agg_mode,
        "h14q03": agg_mode,
        "h14q04": agg_sum,
        "h14q05": agg_sum
    }).reset_index()
    asset_agg.rename(columns={
        "h14q02": "asset_type",
        "h14q03": "asset_owned",
        "h14q04": "asset_quantity",
        "h14q05": "asset_value"
    }, inplace=True)
    return asset_agg

def aggregate_assets_15(df_asset):
    # 2015: GSEC14: h14q2, h14q3, h14q4, h14q5
    asset_agg = df_asset.groupby("hhid").agg({
        "h14q2": agg_mode,
        "h14q3": agg_mode,
        "h14q4": agg_sum,
        "h14q5": agg_sum
    }).reset_index()
    asset_agg.rename(columns={
        "h14q2": "asset_type",
        "h14q3": "asset_owned",
        "h14q4": "asset_quantity",
        "h14q5": "asset_value"
    }, inplace=True)
    return asset_agg

def aggregate_assets_13(df_asset):
    # 2013: GSEC14A: h14q2, h14q3, h14q4, h14q5
    asset_agg = df_asset.groupby("hhid").agg({
        "h14q2": agg_mode,
        "h14q3": agg_mode,
        "h14q4": agg_sum,
        "h14q5": agg_sum
    }).reset_index()
    asset_agg.rename(columns={
        "h14q2": "asset_type",
        "h14q3": "asset_owned",
        "h14q4": "asset_quantity",
        "h14q5": "asset_value"
    }, inplace=True)
    return asset_agg



df_2019_edu_agg = aggregate_education_18_19(df_2019_gsec4)
df_2019_inc_agg = aggregate_income_18_19(df_2019_gsec7_2)
df_2019_asset_agg = aggregate_assets_18_19(df_2019_gsec14)
merged_2019 = df_2019_gsec1.merge(df_2019_edu_agg, on="hhid", how="left")
merged_2019 = merged_2019.merge(df_2019_inc_agg, on="hhid", how="left")
merged_2019 = merged_2019.merge(df_2019_asset_agg, on="hhid", how="left")
merged_2019 = merged_2019.merge(df_2019_pov, on="hhid", how="left")
merged_2019["year"] = 2019

# 2018
df_2018_edu_agg = aggregate_education_18_19(df_2018_gsec4)
df_2018_inc_agg = aggregate_income_18_19(df_2018_gsec7_2)  
df_2018_asset_agg = aggregate_assets_18_19(df_2018_gsec14)
merged_2018 = df_2018_gsec1.merge(df_2018_edu_agg, on="hhid", how="left")
merged_2018 = merged_2018.merge(df_2018_inc_agg, on="hhid", how="left")
merged_2018 = merged_2018.merge(df_2018_asset_agg, on="hhid", how="left")
merged_2018 = merged_2018.merge(df_2018_pov, on="hhid", how="left")
merged_2018["year"] = 2018

# 2015
df_2015_edu_agg = aggregate_education_15_13(df_2015_gsec4)
df_2015_inc_agg = aggregate_income_15(df_2015_gsec11_2)
df_2015_asset_agg = aggregate_assets_15(df_2015_gsec14)
merged_2015 = df_2015_gsec1.merge(df_2015_edu_agg, on="hhid", how="left")
merged_2015 = merged_2015.merge(df_2015_inc_agg, on="hhid", how="left")
merged_2015 = merged_2015.merge(df_2015_asset_agg, on="hhid", how="left")
merged_2015 = merged_2015.merge(df_2015_pov, on="hh", how="left")
merged_2015["year"] = 2015

# 2013
df_2013_gsec1.rename(columns={'HHID': 'hhid'}, inplace=True)
df_2013_gsec4.rename(columns={'HHID': 'hhid'}, inplace=True)
df_2013_gsec11_A.rename(columns={'HHID': 'hhid'}, inplace=True)
df_2013_gsec14A.rename(columns={'HHID': 'hhid'}, inplace=True)
df_2013_pov.rename(columns={'HHID': 'hhid'}, inplace=True)



df_2013_edu_agg = aggregate_education_15_13(df_2013_gsec4)
df_2013_inc_agg = aggregate_income_13(df_2013_gsec11_A)
df_2013_asset_agg = aggregate_assets_13(df_2013_gsec14A)
merged_2013 = df_2013_gsec1.merge(df_2013_edu_agg, on="hhid", how="left")
merged_2013 = merged_2013.merge(df_2013_inc_agg, on="hhid", how="left")
merged_2013 = merged_2013.merge(df_2013_asset_agg, on="hhid", how="left")
merged_2013 = merged_2013.merge(df_2013_pov, on="hhid", how="left")
merged_2013["year"] = 2013

for df in [merged_2019, merged_2018, merged_2015, merged_2013]:
    if "region_x" in df.columns:
        df["region"] = df.get("region_x", pd.NA)
    if "regurb_x" in df.columns:
        df["regurb"] = df.get("regurb_x", pd.NA)
    if "district_x" in df.columns:
        df["district"] = df.get("district_x", pd.NA)
    if "urban_x" in df.columns:
        df["urban"] = df.get("urban_x", pd.NA)
    if "subreg_x" in df.columns:
        df["subregion"] = df.get("subreg_x", pd.NA)

final_cols = [
    "hhid", "year",
    # Education
    "attended_school", "reason_no_school", "can_read_write",
    "highest_grade", "reason_left_school", "funding_source",
    # Income
    "income_source_code", "income_received", "income_cash", "income_in_kind",
    # Assets
    "asset_type", "asset_owned", "asset_quantity", "asset_value",
    # Geography
    "region", "regurb", "subregion", "district", "urban",
    # Household characteristics from POV
    "hsize", "equiv", "nrrexp30", "cpexp30", "welfare", "plinen", "spline"
]


def ensure_key_columns(df, cols):
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols]

df_2019_final = ensure_key_columns(merged_2019.copy(), final_cols)
df_2018_final = ensure_key_columns(merged_2018.copy(), final_cols)
df_2015_final = ensure_key_columns(merged_2015.copy(), final_cols)
df_2013_final = ensure_key_columns(merged_2013.copy(), final_cols)

# ----- Vertically stack the four waves (each now one record per household) -----
df_all_final = pd.concat([df_2019_final, df_2018_final, df_2015_final, df_2013_final], ignore_index=True)

# Save the final combined aggregated dataset.
df_all_final.to_csv(r"..\Datasets\Uganda\combined_key_variables_aggregated.csv", index=False)

# %%

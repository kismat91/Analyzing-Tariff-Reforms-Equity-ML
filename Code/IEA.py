#%% Import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

#%%
iea_consumption_file = r"..\Datasets\IEA data\International Energy Agency - Electricity consumption per capita, Uganda.csv"
iea_final_sector_file = r"..\Datasets\IEA data\International Energy Agency - electricity final consumption by sector in Uganda.csv"
iea_final_sector_2022_file = r"..\Datasets\IEA data\International Energy Agency - Electricity final consumption by sector, Uganda, 2022.csv"

df_iea_consumption = pd.read_csv(iea_consumption_file)
df_iea_final_sector = pd.read_csv(iea_final_sector_file)
df_iea_final_sector_2022 = pd.read_csv(iea_final_sector_2022_file)

#data
print("IEA Electricity Consumption per Capita (Uganda):")
print(df_iea_consumption.shape)
print(df_iea_consumption.head())
print("\nIEA Electricity Final Consumption by Sector (Uganda):")
print(df_iea_final_sector.shape)
print(df_iea_final_sector.head())
print("\nIEA Electricity Final Consumption by Sector, Uganda 2022:")
print(df_iea_final_sector_2022.shape)
print(df_iea_final_sector_2022.head())
# %% Pre Process

print("Missing values in IEA consumption dataset:")
print(df_iea_consumption.isnull().sum())

if 'Year' in df_iea_consumption.columns:
    df_iea_consumption['Year'] = pd.to_numeric(df_iea_consumption['Year'], errors='coerce')
for df in [df_iea_final_sector, df_iea_final_sector_2022]:
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

df_iea_consumption.fillna(method='ffill', inplace=True)
df_iea_final_sector.fillna(method='ffill', inplace=True)
df_iea_final_sector_2022.fillna(method='ffill', inplace=True)


# %%
# electricity consumption per capita over the years
if 'Year' in df_iea_consumption.columns and 'Electricity consumption per capita, Uganda' in df_iea_consumption.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_iea_consumption, x='Year', y='Electricity consumption per capita, Uganda', marker="o")
    plt.title("Electricity Consumption per Capita in Uganda")
    plt.xlabel("Year")
    plt.ylabel("Consumption per Capita (kWh)")
    plt.tight_layout()
    plt.show()
# %%
# consumption over the years for each sector
if 'Year' in df_iea_final_sector.columns and 'electricity final consumption by sector in Uganda' in df_iea_final_sector.columns:
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_iea_final_sector, x='Year', y='Value', hue='electricity final consumption by sector in Uganda', marker="o")
    plt.title("Electricity Final Consumption by Sector in Uganda Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Consumption (GWh or appropriate units)")
    plt.legend(title='Sector')
    plt.tight_layout()
    plt.show()

# sector consumption for the year 2022 
plt.figure(figsize=(10, 6))
sns.barplot(data=df_iea_final_sector_2022, x='Electricity final consumption by sector, Uganda, 2022', y='Value')
plt.title("Electricity Final Consumption by Sector in Uganda (2022)")
plt.xlabel("Sector")
plt.ylabel("Consumption (GWh or appropriate units)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# %%

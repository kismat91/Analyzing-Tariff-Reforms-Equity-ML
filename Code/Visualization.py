import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r'C:\Users\aroub\Desktop\Book7.csv')

df.columns = df.columns.str.strip().str.replace(" +", " ", regex=True)

columns = ["Domestic", "Commercial", "Medium Industries", "Large Industries", "Extra Large"]

df[columns] = df[columns].replace({",": ""}, 
                                  regex=True).apply(pd.to_numeric, 
                                                    errors='coerce')

df1 = df.melt(id_vars=["Year", "Quarter"], 
              var_name="Category", 
              value_name="Values")

quarter_order = ["Q1", "Q2", "Q3", "Q4"]

df1["Quarter"] = pd.Categorical(df1["Quarter"], 
                                categories=quarter_order, 
                                ordered=True)

df1 = df1.sort_values(["Year", "Quarter"], 
                      ascending=[False, True])

plt.figure(figsize=(14, 6))

sns.barplot(data=df1, 
            x="Category", 
            y="Values",
            hue="Year", ci=None)

plt.xlabel("Category")
plt.ylabel("Values")

plt.title("Different Categories Across Years")

plt.xticks(rotation=45)

plt.legend(title="Year", 
           bbox_to_anchor=(1, 1))

plt.show()
#-------------------------------------------------------------
#-------------------------------------------------------------

df2 = df.melt(id_vars=["Year", "Quarter"], var_name="Category", value_name="Values")

plt.figure(figsize=(10, 5))
plt.bar(df2["Category"], 
        df2["Values"], 
        color=['blue', 'orange', 'green', 'red', 'purple'])

plt.xlabel("Category")
plt.ylabel("Values")
plt.title("Distribution of Different Categories in Q3 2024")
plt.xticks(rotation=45)
plt.show()

#-------------------------------------------------------------
#-------------------------------------------------------------

df = pd.read_csv(r'C:\Users\aroub\Desktop\Capstone Project\Book41.csv')

df.columns = df.columns.str.strip()

df["Year"] = pd.to_numeric(df["Year"],
                           errors='coerce').astype('Int64')  

df["Commercial and Public Services"] = pd.to_numeric(df["Commercial and Public Services"], 
                                                     errors='coerce')

df["Residential"] = pd.to_numeric(df["Residential"], 
                                  errors='coerce')

df["Industry Sector"] = pd.to_numeric(df["Industry Sector"], 
                                      errors='coerce')

df = df.dropna()
df = df.sort_values(by="Year")

plt.figure(figsize=(10, 5))

plt.plot(df["Year"],
         df["Commercial and Public Services"],
         marker='o', 
         linestyle='-', 
         label="Commercial and Public Services")

plt.plot(df["Year"], 
         df["Residential"],
         marker='s', 
         linestyle='--', 
         label="Residential")


plt.plot(df["Year"], 
         df["Industry Sector"],
         marker='^', 
         linestyle='-', 
         label="Industry Sector")

plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Commercial, Residential, and Industry Sector Trends")

plt.show()

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


df = pd.read_csv(r'C:\Users\aroub\Desktop\Capstone Project\Book41.csv')

col = ["Domestic", "Commercial", "Medium Industrial", "Large Industrial"]

df_filtered = df[df["YEAR"] == 2024] 
values = df_filtered[col].sum() 

data = {
    "Category": ["Domestic", "Commercial", "Medium Industrial", "Large Industrial"],
    "Value": values.values
}
df_vis = pd.DataFrame(data)

plt.figure(figsize=(8, 5))

plt.bar(df_vis["Category"], 
        df_vis["Value"],
        color=["blue", "orange", "green", "red"])

plt.title("Energy Consumption by Sector (2024)")

plt.ylabel("Value")
plt.xlabel("Sector")

plt.xticks(rotation=30)

plt.grid(axis="y", 
         linestyle="--",
         alpha=0.7)
plt.show()

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

df = pd.read_csv(r'C:\Users\aroub\Desktop\Capstone Project\Book41.csv')

df.columns = df.columns.str.strip()
sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
sns.lineplot(x='Year',
             y='Value',
             hue='Sector',
             data=df,
             marker='o',
             palette='Blues_d')

plt.title('Electricity Final Consumption by Sector in Uganda', 
          fontsize=14)

plt.xlabel('Year', 
           fontsize=12)

plt.ylabel('Consumption Value', 
           fontsize=12)

plt.xticks(rotation=45) 
plt.legend(title='Sector')  
plt.show()

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

df = pd.read_csv(r'C:\Users\aroub\Desktop\Capstone Project\Book41.csv')

df.columns = df.columns.str.strip()


df = df.dropna(subset=["Year"])  
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")  
df = df.dropna(subset=["Year"])  
df["Year"] = df["Year"].astype(int)  

col_name = "Electricity consumption per capita, Uganda"
df[col_name] = df[col_name].astype(str).str.replace(",", "").astype(float)


df = df.sort_values(by="Year")


plt.figure(figsize=(10, 6))
plt.plot(df["Year"],
         df[col_name],
         marker="o", 
         linestyle="-", 
         linewidth=2, color="b")

plt.xlabel("Year", 
           fontsize=12)

plt.ylabel("Electricity Consumption per Capita", 
           fontsize=12)

plt.title("Electricity Consumption Per Capita in Uganda Over Time", 
          fontsize=14)

plt.xticks(rotation=45, 
           fontsize=10)

plt.grid(True,
         linestyle="--", 
         alpha=0.7)
plt.show()

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

df.columns = df.columns.str.strip()
df["Quarter"] = df["Quarter"].astype(str).str.strip().fillna("Unknown")

df["Year"] = df["Year"].astype(str)

df["Time"] = df["Year"] + " " + df["Quarter"]


numeric_columns = ["Domestic", "Commercial", "Medium Industries", "Large Industries", "Streetlighting"]

for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(",", "").replace("-", np.nan).astype(float)


df = df.sort_values(by=["Year", "Quarter"], ascending=[True, False])

sns.set_style("whitegrid")

plt.figure(figsize=(12, 6))

for column in numeric_columns:
    plt.plot(df["Time"], 
             df[column], 
             marker="o", 
             label=column,
             linestyle='-',
             linewidth=2)

plt.xlabel("Time (Year and Quarter)",
           fontsize=12)

plt.ylabel("Values", 
           fontsize=12)

plt.title("Energy Consumption by Sector Over Time (UMEME UGANDA LIMITED)",
          fontsize=14)

plt.xticks(rotation=45, 
           fontsize=10)

plt.legend(title="Sector", 
           fontsize=10)

plt.grid(True, 
         linestyle='--', 
         alpha=0.7)

plt.show()


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

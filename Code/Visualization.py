import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\aroub\Desktop\Book7.csv')

df.columns = df.columns.str.strip().str.replace(" +", " ", regex=True)

columns = ["Domestic", "Commercial", "Medium Industries", "Large Industries", "Extra Large"]
df[columns] = df[columns].replace({",": ""}, regex=True).apply(pd.to_numeric, errors='coerce')

df1 = df.melt(id_vars=["Year", "Quarter"], var_name="Category", value_name="Values")

quarter_order = ["Q1", "Q2", "Q3", "Q4"]
df1["Quarter"] = pd.Categorical(df1["Quarter"], categories=quarter_order, ordered=True)
df1 = df1.sort_values(["Year", "Quarter"], ascending=[False, True])

plt.figure(figsize=(14, 6))
sns.barplot(data=df1, x="Category", y="Values", hue="Year", ci=None)

plt.xlabel("Category")
plt.ylabel("Values")
plt.title("Different Categories Across Years")
plt.xticks(rotation=45)
plt.legend(title="Year", bbox_to_anchor=(1, 1))

plt.show()
#-------------------------------------------------------------
#-------------------------------------------------------------

columns = ["Domestic", "Commercial", "Medium Industries", "Large Industries", "Extra Large"]
df[columns] = df[columns].replace({",": ""}, regex=True).apply(pd.to_numeric, errors='coerce')

df2 = df.melt(id_vars=["Year", "Quarter"], var_name="Category", value_name="Values")

plt.figure(figsize=(10, 5))
plt.bar(df2["Category"], df2["Values"], color=['blue', 'orange', 'green', 'red', 'purple'])

plt.xlabel("Category")
plt.ylabel("Values")
plt.title("Distribution of Different Categories in Q3 2024")
plt.xticks(rotation=45)
plt.show()

#-------------------------------------------------------------
#-------------------------------------------------------------

df = pd.read_csv(r'C:\Users\aroub\Desktop\Capstone Project\Book41.csv')

df.columns = df.columns.str.strip()

# Convert columns to numeric (ignoring errors to handle any non-numeric values)
df["Year"] = pd.to_numeric(df["Year"], errors='coerce').astype('Int64')  # Convert Year to integer type
df["Commercial and Public Services"] = pd.to_numeric(df["Commercial and Public Services"], errors='coerce')
df["Residential"] = pd.to_numeric(df["Residential"], errors='coerce')
df["Industry Sector"] = pd.to_numeric(df["Industry Sector"], errors='coerce')

df = df.dropna()
df = df.sort_values(by="Year")

plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["Commercial and Public Services"], marker='o', linestyle='-', label="Commercial and Public Services")
plt.plot(df["Year"], df["Residential"], marker='s', linestyle='--', label="Residential")
plt.plot(df["Year"], df["Industry Sector"], marker='^', linestyle='-', label="Industry Sector")

plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Commercial, Residential, and Industry Sector Trends")

plt.show()

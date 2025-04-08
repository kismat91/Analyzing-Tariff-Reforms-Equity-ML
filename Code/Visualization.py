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

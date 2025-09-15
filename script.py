# --------------------------------------------------
# Python Assignment
# Student Name: [Your Name]
# Task: Data Analysis and Visualization (Iris Dataset)
# --------------------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set seaborn style for prettier plots
sns.set(style="whitegrid")

# Open report file
report = open("report.txt", "w")

# --------------------
# Task 1: Load and Explore Dataset
# --------------------
print("\n================ TASK 1: Load and Explore Dataset ================\n")
report.write("\n================ TASK 1: Load and Explore Dataset ================\n\n")

try:
    # Load dataset
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
except FileNotFoundError:
    print("Error: Dataset file not found.")
    report.write("Error: Dataset file not found.\n")
    exit()
except ValueError as e:
    print("Error while loading dataset:", e)
    report.write("Error while loading dataset: " + str(e) + "\n")
    exit()

# Display first few rows
print(df.head())
report.write("First 5 rows of dataset:\n")
report.write(str(df.head()) + "\n\n")

# Dataset shape
print(df.shape)
report.write("Dataset shape (rows, columns):\n")
report.write(str(df.shape) + "\n\n")

# Info (cannot capture fully, note in report)
print(df.info())
report.write("Dataset info: (see console for details)\n\n")

# Summary stats
print(df.describe())
report.write("Summary statistics:\n")
report.write(str(df.describe()) + "\n\n")

# Count of each species
print(df['species'].value_counts())
report.write("Unique species and counts:\n")
report.write(str(df['species'].value_counts()) + "\n\n")


# --------------------
# Task 2: Basic Analysis
# --------------------
print("\n================ TASK 2: Basic Analysis ================\n")
report.write("\n================ TASK 2: Basic Analysis ================\n\n")

# Average petal length
avg_petal_length = df.groupby('species')['petal length (cm)'].mean()
print(avg_petal_length)
report.write("Average Petal Length per Species:\n")
report.write(str(avg_petal_length) + "\n\n")

# Species with max sepal width
max_species = df.groupby('species')['sepal width (cm)'].mean().idxmax()
print(f"Species with highest avg sepal width: {max_species}")
report.write(f"Species with highest avg sepal width: {max_species}\n\n")


# --------------------
# Task 3: Data Visualization
# --------------------
print("\n================ TASK 3: Data Visualization ================\n")
report.write("\n================ TASK 3: Data Visualization ================\n\n")

# 1. Line Chart
plt.figure(figsize=(8,5))
plt.plot(df['petal length (cm)'], label="Petal Length")
plt.title("Line Chart of Petal Length")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.savefig("line_chart_petal_length.png")
plt.show()

# 2. Bar Chart
plt.figure(figsize=(8,5))
avg_petal_length.plot(kind='bar', color=['red','green','blue'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.savefig("bar_chart_avg_petal_length.png")
plt.show()

# 3. Histogram
plt.figure(figsize=(8,5))
plt.hist(df['sepal width (cm)'], bins=20, color='skyblue', edgecolor='black', label="Sepal Width")
plt.title("Histogram of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("histogram_sepal_width.png")
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.savefig("scatterplot_sepal_vs_petal.png")
plt.show()


# --------------------
# Task 4: Observations
# --------------------
print("\n================ TASK 4: Observations ================\n")
report.write("\n================ TASK 4: Observations ================\n\n")

observations = """
1. The line chart shows how petal length varies across samples. 
   There are clear differences between species.

2. The bar chart reveals that Iris-virginica has the largest 
   average petal length, while Iris-setosa has the smallest.

3. The histogram shows sepal width is most common between 
   2.5 cm and 3.5 cm, with fewer very wide or very narrow sepals.

4. The scatter plot indicates clear separation between species 
   when comparing sepal length and petal length.
"""
print(observations)
report.write(observations + "\n")

report.close()
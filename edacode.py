# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('titanic.csv')  # Make sure the dataset is in the same directory

# Display the first few rows of the dataset
df.head()
# Generate summary statistics
summary_stats = df.describe()
print("Summary Statistics:")
print(summary_stats)
# Create histograms for numeric features
plt.figure(figsize=(10, 5))
df['Age'].hist(bins=30, color='blue', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()
# Create boxplots for numeric features
plt.figure(figsize=(10, 5))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Boxplot of Age by Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()
# Use pairplot to visualize relationships between features
sns.pairplot(df, hue='Survived')
plt.title('Pairplot of Titanic Dataset')
plt.show()
print(df.head())
print(df.dtypes)
# Identify patterns, trends, or anomalies
# Example: Check for missing values
missing_values = df.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)

# Step 1: Remove or Convert Non-Numeric Columns
# Option 1: Drop non-numeric columns
df_numeric = df.select_dtypes(include=['number'])
# Step 2: Create a correlation matrix
plt.figure(figsize=(12, 8))
correlation = df_numeric.corr()  # Use the numeric DataFrame
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

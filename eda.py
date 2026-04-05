import seaborn as sns
import matplotlib.pyplot as plt

def inspect_structure(df):
    """
    Step 2: Inspect Dataset Structure
    """
    df.info()

def visualize_outliers(df, column_name):
    """
    Step 7: Detect Outliers in Numerical Data
    """
    print(df.describe())
    if column_name in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column_name])
        plt.title(f'{column_name.capitalize()} Distribution')
        plt.savefig(f'{column_name}_boxplot.png')
        print(f"Saved boxplot of '{column_name}' to '{column_name}_boxplot.png'.")
        plt.close()

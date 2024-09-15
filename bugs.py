import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
arr1 = np.array([1, 2,3])
arr2 = np.array([1, 2, 3])

# Attempting an invalid operation between arrays of different shapes
result = arr1 + arr2  # ValueError: operands could not be broadcast together

arr = np.array([1, 2, 3])

# Trying to access an invalid index
print(arr[2])  # IndexError: index out of bounds

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Trying to access a non-existent column
print(df['B'])  # KeyError

df = pd.DataFrame({
    'A': [1, 2, 3]
})

# Assigning a list of incorrect length
df['B'] = [10, 20]  # ValueError: Length mismatch

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1]
})

# Modifying a slice of the DataFrame
df_slice = df[df['A'] > 2]
df_slice['C'] = 10  # SettingWithCopyWarning

plt.plot([1, 2, 3])
plt.title(123)  # TypeError: 'value' must be an instance of str or bytes, not int
plt.show()

# DataFrame with a string in numerical columns
df = pd.DataFrame({
    'A': ['1', '2', 'three'],
    'B': [4, 5, 6]
})

# Attempting to plot with invalid data types
sns.scatterplot(x='A', y='B', data=df)  # ValueError: could not convert string to float


# Creating a plot and incorrectly setting tick labels
plt.plot([1, 2, 3])
plt.xticks([1, 2, 3], ['A', 'B', 'C'])  # UserWarning: FixedFormatter should only be used with FixedLocator
plt.show()
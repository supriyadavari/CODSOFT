import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("imdb_movies.csv")

# Rename column
df.rename(columns={'Name': 'Movie'}, inplace=True)

# Fix Duration column
df['Duration'] = df['Duration'].astype(str)
df['Duration'] = df['Duration'].str.replace(' min', '', regex=False)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Fix Rating
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Rating'].fillna(df['Rating'].mean(), inplace=True)

# Best year
print("Best Year:", df.groupby('Year')['Rating'].mean().idxmax())

# Top 5 movies
print(df[['Movie', 'Rating']].sort_values(by='Rating', ascending=False).head())

# Scatter plot
plt.scatter(df['Duration'], df['Rating'])
plt.xlabel("Duration (minutes)")
plt.ylabel("Rating")
plt.show()
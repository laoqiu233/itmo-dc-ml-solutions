import pandas as pd

df = pd.read_csv('data.csv', index_col='region_name')
df.info()

# Представим ситуацию, что из-за невнимательности операциониста, регионы: 
# Республика Алтай, Магаданская обл. оказались не представлены в итоговой сводке.
df.drop(['Республика Алтай', 'Магаданская обл.'], inplace=True)
df.sort_values('salary', inplace=True)

print(df.head(6).tail(1))
print(df.head(49).tail(1))
print(df.head(51).tail(1))

print(df['salary'].mean())
print(df['salary'].median())
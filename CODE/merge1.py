import pandas as pd

csv_file_1 = 'New_features/com_features.csv'
csv_file_2 = 'New_features/final_features.csv'

df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

merged_df = pd.concat([df1, df2], ignore_index=True)


merged_df.to_csv('merged2.csv', index=False)

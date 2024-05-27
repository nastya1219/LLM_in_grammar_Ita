import pandas as pd

# Загрузка данных из файлов TSV
data1 = pd.read_csv('Hermes_ItaCoLA_Ita-prompt_12.04.2024_label.tsv', sep='\t')
data2 = pd.read_csv('Hermes_ItaCoLA_Ita-prompt_12.04.2024_label_581-946.tsv', sep='\t', skiprows=2)

# Склеиваем датасеты по столбцу 'column_name'
result = pd.concat([data1, data2], axis=1)

print(result)

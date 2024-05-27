from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd

matches_values = []
label_corr_result_model_list = []
label_corr_result_model = pd.read_csv('mistralai-Mixtral-8x7B-Instruct-v0.1_label_only_promt_italian_10.04.2024.tsv', sep='\t')
for i in (label_corr_result_model['label']):
    label_corr_result_model_list.append(int(i))
Ita = pd.read_csv('ItaCoLA_dataset.tsv', sep='\t')
for i in (Ita['UniqueIndexID']-1):
    if Ita['Split'][i] == 'dev':
        matches_values += [Ita['Acceptability'][i]]

y_true = [int(i) for i in label_corr_result_model_list] # то, что что дано в датасете
y_pred = [int(i) for i in matches_values] # то, что предсказывает модель

report = classification_report(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print(report)

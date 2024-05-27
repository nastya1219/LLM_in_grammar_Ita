from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd

matches_values = []
label_result_model_list = []
label_result_model = pd.read_csv('mistralai-Mixtral-8x7B-Instruct-v0.Ita.10.04.2024_label-explanation-correction.tsv', sep='\t')
for i in (label_result_model['label']):
    label_result_model_list.append(int(i))
Ita = pd.read_csv('ItaCoLA_dataset.tsv', sep='\t')
for i in (Ita['UniqueIndexID']-1):
    if Ita['Split'][i] == 'dev':
        matches_values += [Ita['Acceptability'][i]]

y_true = [int(i) for i in label_result_model_list] # то, что что дано в датасете
y_pred = [int(i) for i in matches_values] # то, что предсказывает модель

report = classification_report(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print(mcc)

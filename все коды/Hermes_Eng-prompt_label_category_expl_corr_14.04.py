from hugchat import hugchat
from hugchat.login import Login
import pandas as pd
import time
from tqdm import tqdm
import csv
import re


class HuggingChat:
    """
    API for accessing HuggingChat
    List of models: https://huggingface.co/chat
    Requires email and password from https://huggingface.co to use
    """

    def __init__(
        self,
        email: str,
        password: str,
        system_prompt: str = "",
        cookie_path_dir: str = "./cookies_snapshot",
        model: str = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    ):
        self.sign = Login(email, password)
        cookies = self.sign.login()
        self.sign.saveCookiesToDir(cookie_path_dir)
        self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        # Check actual list of models on https://huggingface.co/chat/settings
        self.models = {
            "CohereForAI/c4ai-command-r-plus": 0,
            "meta-llama/Meta-Llama-3-70B-Instruct": 1,
            "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1": 2,
            "mistralai/Mixtral-8x7B-Instruct-v0.1": 3,
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 4,
            "google/gemma-1.1-7b-it": 5,
            "mistralai/Mistral-7B-Instruct-v0.2": 6,
            "microsoft/Phi-3-mini-4k-instruct": 7,
        }
        self.system_prompt = system_prompt
        self.model = self.models[model]
        self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict(), system_prompt=self.system_prompt)
        self.chatbot.switch_llm(self.model)
        self.chatbot.new_conversation(switch_to=True, system_prompt=self.system_prompt)

    def prompt(self, prompt: str) -> str:
        return str(self.chatbot.query(prompt))

    def delete_conversations(self) -> None:
        """
        Deletes all conversations in a user's profile
        """
        self.chatbot.delete_all_conversations()
        self.chatbot.new_conversation(switch_to=True,system_prompt=self.system_prompt)

    def switch_model(self, model: str) -> None:
        self.model = self.models[model]
        self.chatbot.switch_llm(self.model)
        self.chatbot.new_conversation(switch_to=True,system_prompt=self.system_prompt)

    def switch_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

# Работаем с датасетом

answers = []
sentences = []
Ita = pd.read_csv('ItaCoLA_dataset.tsv', sep='\t')
for i in (Ita['UniqueIndexID']-1):
    if Ita['Split'][i] == 'dev':
        answers += [Ita['Acceptability'][i]]
        sentences += [Ita['Sentence'][i]]


# Создаем свой файл
# Список заголовков столбцов
headers = ['label', 'category', 'explanation', 'correction']

# Имя файла для записи
filename = "Hermes_label-category-expl-correction_01.05.2024_2.0.tsv"

# Запись заголовков столбцов в файл
with open(filename, 'w', encoding="utf-8", newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(headers)

mistral_hugginchat = HuggingChat("Marina78", "Marina78#", model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                                 system_prompt="Sei un assistente utile. Il tuo compito è classificare se la frase è grammaticalmente corretta / accettabile. Risposta 1, se la frase è corretta, risposta 0, se la frase non è corretta. Il tuo output dovrebbe essere simile a questo: in primo luogo, scrivi 'Label:' e restituisce l'etichetta (0/1), quindi scrivi 'Category:' e scrivi il tipo di errore (Morphology, Syntax, Semantics), se la frase non è corretta. Se la frase è corretta, restituisci 'Correct'. Quindi scrivi 'Explanation:' e spiega la tua risposta. Infine, scrivi 'Correct Sentence:' e restituisce la frase corretta (se la frase data non era corretta) o la frase 'La frase è già corretta.'(se la frase data era corretta). Un esempio di output: Label: 0 Category: Syntax Explanation: La frase non è corretta perché ci deve essere ' so 'invece di' e ' perché i soggetti Masha e Jane sono plurali. Correct Sentence: Masha e Jane sono belle.")

for sentence in tqdm(sentences):
    label = "Nothing"
    explanation = "Nothing"
    category = "Nothing"
    correction = 'Nothing'
    while label == "Nothing" or explanation == "Nothing" or category == "Nothing" or correction == 'Nothing':
        try:
            time.sleep(8)
            data = mistral_hugginchat.prompt(sentence)
            data = data.replace('\n\n', ' ')
            data = data.replace('\n', ' ')
            print(data)
            label_match = re.search(r'Label:\s*(\d+)', data)
            category_match = re.search(r'Category:\s*(.+)', data)
            explanation_match = re.search(r'Explanation:\s*(.+)', data)
            correction_match = re.search(r'Correct Sentence:\s*(.+)', data)

            # Извлекаем значения из найденных совпадений
            label = label_match.group(1).strip()
            category = category_match.group(1).strip()
            explanation = explanation_match.group(1).strip()
            correction = correction_match.group(1).strip()

        except Exception as e:
            print(e)
            label = "Nothing"
            explanation = "Nothing"
            category = "Nothing"
            correction = 'Nothing'
            try:
                time.sleep(4)
                mistral_hugginchat.delete_conversations()
            except Exception as e:
                try:
                    time.sleep(4)
                    mistral_hugginchat.delete_conversations()
                except Exception as e:
                    time.sleep(4)
                    mistral_hugginchat.delete_conversations()

    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"{label}\t{category}t{explanation}t{correction}\n")

    try:
        time.sleep(4)
        mistral_hugginchat.delete_conversations()
    except Exception as e:
        try:
            time.sleep(4)
            mistral_hugginchat.delete_conversations()
        except Exception as e:
            time.sleep(4)
            mistral_hugginchat.delete_conversations()

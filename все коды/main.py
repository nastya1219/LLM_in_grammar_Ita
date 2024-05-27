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
            model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ):
        self.sign = Login(email, password)
        cookies = self.sign.login()
        self.sign.saveCookiesToDir(cookie_path_dir)
        self.chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        # Check actual list of models on https://huggingface.co/chat/settings
        self.models = {
            "mistralai/Mixtral-8x7B-Instruct-v0.1": 0,
            "meta-llama/Llama-2-70b-chat-hf": 1,
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 2,
            "codellama/CodeLlama-34b-Instruct-hf": 3,
            "mistralai/Mistral-7B-Instruct-v0.2": 4,
            "openchat/openchat-3.5-1210": 5,
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
        self.chatbot.new_conversation(switch_to=True, system_prompt=self.system_prompt)

    def switch_model(self, model: str) -> None:
        self.model = self.models[model]
        self.chatbot.switch_llm(self.model)
        self.chatbot.new_conversation(switch_to=True, system_prompt=self.system_prompt)

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
headers = ['label']

# Имя файла для записи
filename = "mistralai-Mixtral-8x7B-Instruct-v0.1_COLA_English_no-role_09.04.2024_label.tsv"

# Запись заголовков столбцов в файл
with open(filename, 'w', encoding="utf-8", newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(headers)

mistral_hugginchat = HuggingChat("Marina78", "Marina78#", model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                 system_prompt='Your are a helpful assistant. Your task is to classify whether the sentence is grammatically correct/acceptable. Answer 1, if the sentence is correct, answer 0, if the sentence is incorrect. Your output should look like this: Write “Label:” and return the label.')

for sentence in tqdm(sentences):
    label = "Nothing"
    while label == "Nothing":
        try:
            time.sleep(8)
            data = mistral_hugginchat.prompt(sentence)
            data = data.replace('\n\n', ' ')
            data = data.replace('\n', ' ')
            data = data.replace('**', '')
            print(data)
            if "python" in data and "def" in data:
                data = "Nothing"
            label_match = re.search(r'Label:.+(\d+)', data)

            label = label_match.group(1).strip()

        except Exception as e:
            print(e)
            label = "Nothing"
            try:
                time.sleep(2)
                mistral_hugginchat.delete_conversations()
            except Exception as e:
                try:
                    time.sleep(2)
                    mistral_hugginchat.delete_conversations()
                except Exception as e:
                    time.sleep(2)
                    mistral_hugginchat.delete_conversations()

    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"{label}\n")

    try:
        time.sleep(2)
        mistral_hugginchat.delete_conversations()
    except Exception as e:
        try:
            time.sleep(2)
            mistral_hugginchat.delete_conversations()
        except Exception as e:
            time.sleep(2)
            mistral_hugginchat.delete_conversations()

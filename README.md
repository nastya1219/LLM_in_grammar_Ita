В данном исследовании с помощью моделей определялась грамматичность 946 предложений на итальянском языке. Для более полной картины были использованы разные промты и сценарии, благодаря которым модели понимали, ответ какой структуры от них ожидался.

Исследование подобного плана позволит ученым сверять информацию о правильности предложений с помощью больших языковых моделей, а также исправлять ошибки в предложениях и текстах.

Для более точного оценивания работы моделей потребовалось три прогона кодов. Результаты прогонов и рассчеты по ним представлены в данной таблице https://docs.google.com/spreadsheets/d/1J9BBYOGh6G0rZHRwA3wNZReeMDdNCocYjRjTqpBrGNw/edit#gid=2128733731

Рассмотрим сценарии ближе:

•	label: модель определяет правильность предложения и возвращает 1, если, по ее мнению, предложение является правильным, 0 - если неправильным.

•	category: в настоящей работе были выделены 3 категории ошибок: морфология (morphology), синтаксис (syntax), семантика (semantics); от модели ожидалось определение категории, к которой относилась ошибка. Стоит отметить, что именно эти категории были выбраны, потому что именно такие категории используются в датасете RuCoLa.

•	explanation: в данном пункте модель должна была объяснить ошибку, допущенную в предложении, указав место ошибки и классифицировав ее.

•	correction: здесь модели отправлялся запрос об исправлении ошибки, большая языковая модель в аутпуте выдавала исправленное предложение.
В данном репозитории файлы с результатами прогонов разделены на 3 папки. Каждая папка соответствует одному прогону.
В папках файлы в названиях содержат информацию о номере прогона, названии модели, конкретном сценарии и языке, что помогает распознать, к какой части исследования принадлежит файл.

Так как второй и третий раз прогонялись коды только с промтами на итальяснком языке, в папках "второй прогон" и "третий прогон" во всех сценариях, кроме 'label', есть файлы с итальянскими промтами, но нет с английскими.

Коды, с помощью которых отправлялись запросы моделям, все лежат в папке "все коды". Благодаря этим кодам и были получены файлы с ответами модели.

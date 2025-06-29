﻿# neyro_search_for_goods_board

Этот проект представляет собой **микросервис на FastAPI**, который обеспечивает **поиск товаров** по текстовым запросам с использованием **нейросетевой модели** из библиотеки `sentence-transformers`.

Модель `intfloat/multilingual-e5-large` используется для генерации **векторных представлений (embeddings)** описаний товаров, что позволяет находить наиболее релевантные товары даже при отсутствии точных совпадений.

Проект также включает:
- Кэширование эмбеддингов для ускорения работы
- Обработку запросов через REST API
- Поддержку кэша и обновления его раз в сутки
- Управление подключениями к базе данных MySQL
- Логирование времени выполнения и производительности

---

### 🛠 Технологии

| Компонент | Версия / Использование |
|-----------|------------------------|
| Python    | 3.10+                  |
| FastAPI   | 0.68.0                 |
| sentence-transformers | 2.2.2        |
| mysql-connector-python | 2.4.4         |
| torch     | 1.13.1                 |
| uvicorn   | 0.15.0                 |
| pickle    | стандартная библиотека |


---

### 🚀 Запуск

#### 1. Установите зависимости:

```bash
pip install -r requirements.txt
```

#### 2. Запустите сервер:

```bash
uvicorn neyro_search:app --host 0.0.0.0 --port 8000 --reload=False
```

> Сервер будет доступен по адресу: `http://localhost:8000`

---

### 📡 API

#### 🔹 POST `/search`

##### Описание:
Ищет товары по заданному запросу с использованием нейросетевой модели.

##### Параметры:
- `query`: строка, которую нужно искать
- `top_n`: количество возвращаемых результатов (по умолчанию: 20)

##### Пример запроса:

```json
{
  "query": "посудомоечная машина",
  "top_n": 10
}
```

##### Ответ:

```json
{
  "results": [
    {"id": 1, "similarity": 0.94},
    {"id": 3, "similarity": 0.87},
    ...
  ]
}
```

#### 🔹 GET `/health`

##### Описание:
Проверка статуса сервиса.

##### Ответ:

```json
{
  "status": "ok",
  "model": "intfloat/multilingual-e5-large"
}
```

---

### 🧠 Модель

- **Модель**: `intfloat/multilingual-e5-large`
- **Тип**: мультиязычная модель, обученная на большом количестве текстовых данных
- **Цель**: создание векторного представления текста для сравнения и поиска

---

### 🗃️ База данных

Сервис использует **MySQL** для хранения информации о товарах.

#### 🔧 Настройка подключения:
В коде указана конфигурация подключения к базе:

```python
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3333,
        user="root",
        password="admin",
        database="auth_db",
        auth_plugin='mysql_native_password',
        buffered=True
    )
```

> Убедитесь, что у вас есть таблица `posts` с полями `id`, `title`, `description`.

---

### 🧠 Кэширование

- Эмбеддинги товаров кэшируются в файл `product_embeddings_cache.pkl`
- Кэш обновляется **раз в сутки**
- Это позволяет ускорить работу сервиса и снизить нагрузку на модель

---

### 🧪 Тестирование

Вы можете протестировать API с помощью `curl` или Postman:

```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query":"посудомоечная машина", "top_n":10}'
```

---

### 🧾 Логирование

- Выводятся сообщения о загрузке модели и эмбеддингов
- Время выполнения поиска выводится в консоль
- Кэш проверяется на актуальность перед запуском

---

### 🛡 Безопасность

- Защита от пустых запросов
- Используются `HTTPException` для обработки ошибок
- Модель выставлена в режим `eval()` для предотвращения обучения во время поиска

---

### 🧩 Использование

#### 🔹 Для фронтенда:
- Отправляйте POST-запросы на `/search` с нужным запросом
- Получайте список ID товаров, которые наиболее релевантны

#### 🔹 Для других сервисов:
- Можно использовать как часть архитектуры микросервисов
- Интегрируется с другими системами через REST API

---

### 📌 Автор

**Кудрявцев Данил**  
Системный администратор IT-отдела  
Email: mikushkinodanil4@gmail.com

---

### 📝 Лицензия

MIT License

Copyright (c) 2025 Кудрявцев Данил

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

### 📌 Дополнительно

Если ты хочешь:
- Добавить больше моделей
- Реализовать асинхронный поиск
- Добавить ограничение по времени
- Использовать GPU для ускорения

— пиши, я помогу доработать систему под твой сайт 😊

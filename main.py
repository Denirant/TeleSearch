import telebot
import traceback
import speech_recognition as sr
import subprocess
import os
import logging
from telebot import types
from pymongo import MongoClient
import requests
import razdel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

from datetime import datetime



genres_array = [
    {"id": 1, "genre": "триллер"},
    {"id": 2, "genre": "драма"},
    {"id": 3, "genre": "криминал"},
    {"id": 4, "genre": "мелодрама"},
    {"id": 5, "genre": "детектив"},
    {"id": 6, "genre": "фантастика"},
    {"id": 7, "genre": "приключения"},
    {"id": 8, "genre": "биография"},
    {"id": 9, "genre": "фильм-нуар"},
    {"id": 10, "genre": "вестерн"},
    {"id": 11, "genre": "боевик"},
    {"id": 12, "genre": "фэнтези"},
    {"id": 13, "genre": "комедия"},
    {"id": 14, "genre": "военный"},
    {"id": 15, "genre": "история"},
    {"id": 16, "genre": "музыка"},
    {"id": 17, "genre": "ужасы"},
    {"id": 18, "genre": "мультфильм"},
    {"id": 19, "genre": "семейный"},
    {"id": 20, "genre": "мюзикл"},
    {"id": 21, "genre": "спорт"},
    {"id": 22, "genre": "документальный"},
    {"id": 23, "genre": "короткометражка"},
    {"id": 24, "genre": "аниме"},
    {"id": 26, "genre": "новости"},
    {"id": 27, "genre": "концерт"},
    {"id": 28, "genre": "для взрослых"},
    {"id": 29, "genre": "церемония"},
    {"id": 30, "genre": "реальное ТВ"},
    {"id": 31, "genre": "игра"},
    {"id": 32, "genre": "ток-шоу"},
    {"id": 33, "genre": "детский"},
]




def format_duration(minutes):
    hours, remainder = divmod(minutes, 60)
    return f"{hours} часа {remainder} минуты"

BOT_TOKEN = None

def read_bot_token():
    try:
        from config import BOT_TOKEN
    except ImportError:
        pass

    if not BOT_TOKEN:
        BOT_TOKEN = os.environ.get('YOUR_BOT_TOKEN')

    if BOT_TOKEN is None:
        raise ValueError('Token for the bot must be provided (BOT_TOKEN variable)')
    return BOT_TOKEN

BOT_TOKEN = read_bot_token()

r = sr.Recognizer()
bot = telebot.TeleBot(BOT_TOKEN)

mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['telegramDB']
users_collection = db['users']

LOG_FOLDER = '.logs'
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'{LOG_FOLDER}/app.log'
)

logger = logging.getLogger('telegram-bot')
logging.getLogger('urllib3.connectionpool').setLevel('INFO')

@bot.message_handler(commands=['start'])
def start_message(message):
    user_id = message.from_user.id

    existing_user = users_collection.find_one({'user_id': user_id})

    if not existing_user:
        bot.reply_to(message, 'Добро пожаловать в бот по поиску фильмов!')
        new_user = {'user_id': user_id, 'language': 'ru_RU'}
        users_collection.insert_one(new_user)
    else:
        bot.reply_to(message, 'Добро пожаловать в бот по поиску фильмов!')

    # Add two buttons below the input field
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    row1_buttons = [types.KeyboardButton("Watched")]
    markup.row(*row1_buttons)

    # Second row of buttons
    row2_buttons = [types.KeyboardButton("Favorites genres"), types.KeyboardButton("View collections")]
    markup.row(*row2_buttons)


    bot.send_message(user_id, "Выберите действие:", reply_markup=markup)


# Handle the button clicks - in ["Clear", "Watched", "Favorites genres", "View collections"]
@bot.message_handler(func=lambda message: message.text)
def handle_buttons(message):
    user_id = message.from_user.id

    if message.text == "Clear":
        # Clear the chat
        bot.send_message(user_id, "Chat cleared!")
    elif message.text == "Watched":
        # Retrieve and display the list of watched movies
        user = users_collection.find_one({'user_id': user_id})
        watched_movies = user.get('watched', [])
        if watched_movies:
            movies_list = "\n".join([f"{index + 1}. Название: {movie.get('name', '')}, Дата: {movie.get('date_added', '')}" for index, movie in enumerate(watched_movies)])
            bot.send_message(user_id, f"List of Watched Movies:\n{movies_list}")
        else:
            bot.send_message(user_id, "No watched movies.")
    elif message.text == "Favorites genres":
        # Retrieve user's watched movies and genres
        user = users_collection.find_one({'user_id': user_id})
        watched_movies = user.get('watched', [])

        # Extract genres from watched movies
        all_genres = [genre for movie in watched_movies for genre in movie.get('genres', [])]

        # Create a dictionary to count the occurrences of each genre
        genre_counts = {}
        for genre in all_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        # Create a list of unique genres in descending order of occurrence
        unique_genres = sorted(set(all_genres), key=lambda x: genre_counts[x], reverse=True)

        # Display the list of favorite genres with counts
        genres_with_counts = [f"{genre}: {genre_counts[genre]}" for genre in unique_genres]
        bot.send_message(user_id, f"Your Favorite Genres:\n{', '.join(genres_with_counts)}")
    elif message.text == "View collections":
        # Retrieve user's watched movies and genres
        user = users_collection.find_one({'user_id': user_id})
        watched_movies = user.get('watched', [])

        # Extract genres from watched movies
        all_genres = [genre for movie in watched_movies for genre in movie.get('genres', [])]

        # Create a dictionary to count the occurrences of each genre
        genre_counts = {}
        for genre in all_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        # Create a list of unique genres in descending order of occurrence
        unique_genres = sorted(set(all_genres), key=lambda x: genre_counts[x], reverse=True)

        # Display the list of favorite genres with counts
        genres_with_counts = [f"{genre}: {genre_counts[genre]}" for genre in unique_genres]

        # Number of genres to display
        genres_to_display = 4

        # Create a ReplyKeyboardMarkup
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        row1_buttons = []
        # Iterate over the first 'genres_to_display' genres
        for i in range(min(genres_to_display, len(unique_genres))):
            genre = unique_genres[i]
            # Add a button to the markup
            row1_buttons.append(types.KeyboardButton(genre))

        markup.row(*row1_buttons)

        # Add the "Home" button in a new row
        markup.row(types.InlineKeyboardButton("/start"))

        bot.send_message(user_id, "Выберите любую категорию среди ваших любимых жанров:", reply_markup=markup)
    elif message.text == "Home":
        bot.send_message(message.chat.id, "/start")
    else:
        # Display the selected genre
        bot.send_message(user_id, f"Вы выбрали жанр: {message.text}")

        genre_to_id = {genre["genre"]: genre["id"] for genre in genres_array}

        selected_genre = message.text.lower()  # Convert to lowercase for case-insensitivity
        if selected_genre in genre_to_id:
            # Display the selected id
            selected_id = genre_to_id[selected_genre]
            bot.send_message(user_id, f"Вы выбрали жанр с id: {selected_id}")

            # The API endpoint
            url = 'https://kinopoiskapiunofficial.tech/api/v2.2/films'

            # Parameters for the request
            params = {
                'genres': selected_id,
                'order': 'RATING',
                'type': 'ALL',
                'ratingFrom': 0,
                'ratingTo': 10,
                'yearFrom': 1000,
                'yearTo': 3000,
                'page': 1
            }

            # The API key header
            headers = {
                'accept': 'application/json',
                'X-API-KEY': '100bd984-a616-4585-a265-5744bd4bffb5'
            }

            # Sending the GET request
            response = requests.get(url, params=params, headers=headers)

            # Checking the response
            if response.status_code == 200:
                # Do something with the response.json() data
                data = response.json()
                
                # Extracting film details
                films = data.get('items', [])

                # Creating a formatted string for each film
                film_strings = []
                for index, film in enumerate(films, start=1):
                    film_id = film.get('kinopoiskId', '')
                    film_name = film.get('nameRu', '')
                    film_url = f"https://www.kinopoisk.ru/film/{film_id}/"

                    # Creating the formatted string
                    film_string = f"{index}) {film_name}, <a href='{film_url}'>Ссылка</a>"
    
                    # Appending to the list
                    film_strings.append(film_string)

                # Joining the strings with newline characters
                result_string = '\n'.join(film_strings)

                # Sending the message
                bot.send_message(user_id, f"Контент подобранный по жанру <b>{selected_genre}</b>: \n{result_string}", disable_web_page_preview=True, parse_mode='HTML')
            else:
                # Handle the error
                print(f"Error: {response.status_code}")
                print(response.text)

        else:
            bot.send_message(user_id, "Жанр не найден.")





@bot.message_handler(content_types=['voice'])
def voice_handler(message):
    user_id = message.from_user.id
    user = users_collection.find_one({'user_id': user_id})

    file_id = message.voice.file_id
    file = bot.get_file(file_id)

    file_size = file.file_size
    if int(file_size) >= 715000:
        bot.send_message(message.chat.id, 'Upload file size is too large.')
    else:
        download_file = bot.download_file(file.file_path)
        with open('audio.ogg', 'wb') as file:
            file.write(download_file)

        text = voice_recognizer(user['language'])
        command, content = extract_command_and_content(text)

        if command and content:
            user['last_command'] = command
            user['last_content'] = content
            users_collection.update_one({'user_id': user_id}, {'$set': user})

            response_message = f"Команда: {command}, Название: {content}"
            bot.send_message(user_id, response_message)

            markup = types.InlineKeyboardMarkup()
            markup.add(types.InlineKeyboardButton("Верно", callback_data="correct"),
                       types.InlineKeyboardButton("Неверно", callback_data="incorrect"))
            bot.send_message(user_id, "Это расшифровка верна?", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data in ["correct", "incorrect"])
def transcription_callback_handler(call):
    user_id = call.from_user.id
    user = users_collection.find_one({'user_id': user_id})

    # Check if the user was waiting for confirmation
    if 'last_command' in user and 'last_content' in user:
        if call.data == "correct":
            command = user['last_command']
            content = user['last_content']

            if command == "найди":
                movies = search_movies(content)
                send_movies_results(bot, user_id, movies, command, content)
            else:
                bot.send_message(user_id, f"Команда: {command}, Название: {content} - выполнена.")

        elif call.data == "incorrect":
            bot.send_message(user_id, "Пожалуйста, повторите команду.")
        
        # Clear the last command and content
        users_collection.update_one({'user_id': user_id}, {'$unset': {'last_command': '', 'last_content': ''}})
    else:
        bot.send_message(user_id, "Неверный запрос.")

def add_punctuation_russian(text):
    if not text:
        return ""

    model_name = "arshad-bert-base-uncased-sentence"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    restored_text = ""
    for token, pred_label in zip(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), predictions[0]):
        if token.startswith("##"):
            restored_text = restored_text.rstrip() + token[2:]
        else:
            if pred_label == 1:
                restored_text = restored_text.rstrip() + token + " "
            else:
                restored_text += token + " "

    return restored_text.strip()

def voice_recognizer(language):
    subprocess.run(['ffmpeg', '-i', 'audio.ogg', 'audio.wav', '-y'])
    text = 'Words not recognized.'
    file = sr.AudioFile('audio.wav')
    with file as source:
        try:
            audio = r.record(source)
            text = r.recognize_google(audio, language=language)
            text = add_punctuation_russian(text)
        except:
            logger.error(f"Exception:\n {traceback.format_exc()}")

    return text

def search_movies(query):
    base_url = 'https://kinopoiskapiunofficial.tech/api/v2.2/films'
    api_key = '100bd984-a616-4585-a265-5744bd4bffb5'

    headers = {
        'accept': 'application/json',
        'X-API-KEY': api_key
    }

    params = {
        'order': 'RATING',
        'type': 'ALL',
        'ratingFrom': 0,
        'ratingTo': 10,
        'yearFrom': 1000,
        'yearTo': 3000,
        'keyword': query,
        'page': 1
    }

    response = requests.get(base_url, params=params, headers=headers)

    if response.status_code == 200:
        movies_data = response.json()
        items = movies_data.get('items', [])

        print(query)
        print(items)

        # Извлечение nameRu для каждого фильма
        filtered_movies = [{'name': movie.get('nameRu'), 'id': movie.get('kinopoiskId')} for movie in items if (movie.get('type') == 'FILM' or movie.get('type') == 'TV_SERIES')]

        return filtered_movies
    else:
        print(f"Error: {response.status_code}")
        return None

def send_movies_results(bot, user_id, movies, command, content):
    if not movies:
        bot.send_message(user_id, "По вашему запросу ничего не найдено.")
        return

    user = users_collection.find_one({'user_id': user_id})
    user['movies'] = movies
    user['total_pages'] = (len(movies) + 4) // 5
    user['current_page'] = 1
    users_collection.update_one({'user_id': user_id}, {'$set': user})

    print(movies)

    send_movies_page(bot, user_id, movies, user['current_page'], user['total_pages'])


def send_movies_page(bot, user_id, movies, current_page, total_pages):
    user = users_collection.find_one({'user_id': user_id})
    start_index = (current_page - 1) * 5
    end_index = min(current_page * 5, len(movies))
    page_movies = movies[start_index:end_index]

    buttons = []
    for idx, movie in enumerate(page_movies):
        button_text = movie['name']
        callback_data = f"movie:{movie['id']}"
        buttons.append([types.InlineKeyboardButton(button_text, callback_data=callback_data)])

    navigation_buttons = []
    if current_page > 1:
        navigation_buttons.append([types.InlineKeyboardButton("◀️ Назад", callback_data="prev_page")])
    if current_page < total_pages:
        navigation_buttons.append([types.InlineKeyboardButton("Вперед ▶️", callback_data="next_page")])

    inline_keyboard = buttons + navigation_buttons
    keyboard = types.InlineKeyboardMarkup(inline_keyboard)

    message_text = "Вот что мне удалось найти по вашему запросу:"
    message_text += f"\n\nСтраница {current_page}:"
        

    # Отправляем новое сообщение или редактируем старое
    if 'last_message_id' in user:
        bot.edit_message_text(chat_id=user_id, message_id=user['last_message_id'], text=message_text, reply_markup=keyboard)
    else:
        sent_message = bot.send_message(user_id, message_text, reply_markup=keyboard)
        user['last_message_id'] = sent_message.message_id

def extract_command_and_content(text):
    # Разделяем текст на токены
    tokens = [token.text for token in razdel.tokenize(text.lower())]

    # Ищем ключевые слова и извлекаем команду и контент
    command = None
    content = None
    if "найди" in tokens and len(tokens) > tokens.index("найди") + 1:
        command = "найди"
        content_tokens = tokens[tokens.index("найди") + 1:]
        content = " ".join(content_tokens)
    elif "избранное" in tokens:
        command = "добавь в посмотренно"
        content_tokens = tokens[tokens.index("избранное") + 1:]
        content = " ".join(content_tokens)
    elif "посмотреть" in tokens and "позже" in tokens:
        command = "добавь в посмотреть позже"
        content_tokens = tokens[tokens.index("позже") + 1:]
        content = " ".join(content_tokens)

    return command, content

def get_movie_details(movie_id):
    base_url = f'https://kinopoiskapiunofficial.tech/api/v2.2/films/{movie_id}'
    api_key = '100bd984-a616-4585-a265-5744bd4bffb5'

    headers = {
        'accept': 'application/json',
        'X-API-KEY': api_key
    }

    response = requests.get(base_url, headers=headers)

    if response.status_code == 200:
        film = response.json()

        genres = [genre["genre"] for genre in film['genres']]

        # Формирование текста для бота
        message = f"<b>{film['nameRu']}</b>\n\n"
        message += f"<b>Описание:</b> {film['description']}\n"
        message += f"<b>Год:</b> {film['year']}\n"

        message += f"\n<b>Где посмотреть:\n</b>"
        message += f"Кинопоиск: <a href='{film['webUrl']}'>смотреть</a>\n"

        message += f"\n<b>Длительность:</b> {format_duration(film['filmLength'])}\n"
        message += f"\n<b>Жанры:</b> {', '.join(genres)}"

        movie_name = film['nameRu']

        # Получение URL изображения
        photo_url = film.get('posterUrl') or film.get('posterUrlPreview')

        return message, photo_url, movie_name, genres
    else:
        print(f"Error: {response.status_code}")
        return None

@bot.callback_query_handler(func=lambda call: call.data.startswith("movie:") or call.data.startswith("watched:") or call.data.startswith("unwatched:") or call.data in ["prev_page", "next_page", "return_to_list"])
def movies_callback_handler(call):
    user_id = call.from_user.id
    user = users_collection.find_one({'user_id': user_id})

    print(call.data)

    if call.data == "prev_page":
        user['current_page'] = max(1, user.get('current_page', 1) - 1)
    elif call.data == "next_page":
        user['current_page'] = min(user.get('total_pages', 1), user.get('current_page', 1) + 1)
    elif call.data.startswith("movie:"):
        movie_id = int(call.data.split(":")[1])
        movie_details, photo_url, movie_name, genres = get_movie_details(movie_id)

        # Отправка сообщения с изображением
        bot.send_photo(user_id, photo_url, caption=movie_details, parse_mode='HTML')

        # Создание кнопок
        markup = types.InlineKeyboardMarkup()
        # Проверяем, добавлен ли фильм в просмотренные
        is_watched = any(movie.get('id') == movie_id for movie in user.get('watched', []))

        if is_watched:
            markup.add(types.InlineKeyboardButton("Удалить из просмотренных", callback_data=f"unwatched:{movie_id}"))
            markup.add(types.InlineKeyboardButton("Посмотреть позже", callback_data=f"watch_later:{movie_id}"))
            markup.add(types.InlineKeyboardButton("Вернуться к списку", callback_data="return_to_list"))

        else:
            markup.add(types.InlineKeyboardButton("Добавить в просмотренное", callback_data=f"watched:{movie_id}"))
            markup.add(types.InlineKeyboardButton("Посмотреть позже", callback_data=f"watch_later:{movie_id}"))
            markup.add(types.InlineKeyboardButton("Вернуться к списку", callback_data="return_to_list"))


        # Отправка сообщения с кнопками
        bot.send_message(user_id, "Выберите действие:", reply_markup=markup)
    elif call.data.startswith("watched:"):
        print('watched')

        # Извлекаем id фильма из callback_data
        watched_movie_id = int(call.data.split(":")[1])

        # Ищем фильм с таким id в массиве user['movies']
        watched_movie = next((movie for movie in user.get('movies', []) if movie.get('id') == watched_movie_id), None)

        print(watched_movie)

        if watched_movie:
            # Проверяем, есть ли этот фильм уже в массиве user['watched']
            if 'watched' not in user:
                user['watched'] = []

            # Проверяем, что фильм еще не добавлен в массив user['watched']
            if watched_movie not in user['watched']:
                current_date = datetime.now().strftime("%d.%m.%Y")
                watched_movie['date_added'] = current_date

                movie_details, photo_url, movie_name, genres =  get_movie_details(watched_movie['id'])
                watched_movie['genres'] = genres

                user['watched'].append(watched_movie)
                users_collection.update_one({'user_id': user_id}, {'$set': user})
                bot.send_message(user_id, f"Фильм '{watched_movie['name']}' добавлен в просмотренное ({current_date}).")
            else:
                bot.send_message(user_id, f"Фильм '{watched_movie['name']}' уже находится в просмотренных ({watched_movie.get('date_added', '')}).")
        else:
            bot.send_message(user_id, "Фильм не найден.")
    elif call.data.startswith("unwatched:"):
        # Извлекаем id фильма из callback_data
        movie_id = int(call.data.split(":")[1])

        # Удаляем фильм из массива user['watched']
        user['watched'] = [movie for movie in user.get('watched', []) if movie.get('id') != movie_id]
        users_collection.update_one({'user_id': user_id}, {'$set': user})
        bot.send_message(user_id, f"Фильм удален из просмотренных.")




    users_collection.update_one({'user_id': user_id}, {'$set': user})

    if (not(call.data.startswith("movie:")) and not(call.data.startswith("watched:"))):
        send_movies_page(bot, user_id, user['movies'], user['current_page'], user['total_pages'])

if __name__ == '__main__':
    logger.info('start bot')
    bot.polling(True)
    logger.info('stop bot')

import os

import openai
from openai.embeddings_utils import distances_from_embeddings
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from flask_bootstrap import Bootstrap
from config import Config


def emb_to_list(text):
    parse = text.strip('[]').replace(' ', '').split(',')
    return list(map(float, parse))

def create_context_3ver(question, df, max_len=1500, cont_name = 'content', emb_name = 'embeddings', count_name = 'count', threshold = 0.3):
    """поиск и генерация контекста для вопроса 

    Args:
        question (str): Текст вопроса
        df (pd.DataFrame): База данных с информацией
        max_len (int): Максимальная длина подсказки Defaults to 1800.
        cont_name (str):  название колонки с контентом 
        emb_name (str, optional): Название столбца с ембедингами Defaults to 'embeddings'.
        count_name (str, optional): Название столбца с колличеством токенов элементов promt Defaults to 'count'.
        threshold (float)  максимальное расстояние до подсказки  Defaults to 0.3 (рекомендуется threshold > 0.17 )

    Returns:
        text: текст подсказки для модели, который поможет ответить на вопрос
    """

    help_finde = False #помощь не нашли в данных 

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df[emb_name].values, distance_metric='cosine')

    returns = []
    titles = []
    distance = []
    cur_len = 0
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        if row['distances'] > threshold:
            break

        #print(f"{row['distances']} : {row['title']}")

        cur_len += row[count_name] + 4 # прибавляем токены
        
        #If the context is too long, break
        if cur_len > max_len:
            break
        #print(df.sort_values('distances', ascending=True))
        returns.append(row[cont_name])
        titles.append(row['title'])
        distance.append(int((1-row['distances'])*100))

    # Return the context
    return "\n\n###\n\n".join(returns), titles, distance


def get_answer_from_gpt(QUESTION, information, model):
    if model == 'gpt-3.5-turbo':
        return generate_answer_chat(QUESTION, information)
    else:
        return generate_answer_s(QUESTION, information, model)


def generate_answer_chat(QUESTION, information, mes_add=[], temp = 0, curr_count = 0):
    mes = [
            {"role": "system", "content": "Привет! Ты ассистент помошник по использованию програмного продукта 'Дозор' \
            'Дозор'. Будь вежлив и краток. Здоровайся с каждым пользователем!"},
            {"role": "user", "content": "Главное не отвечай ничего лишнего. Говори только о том, что знаешь."},
            {"role": "user", "content": "Используй эту информацию для обоснования ответа на мой вопрос: " +information},
            {"role": "user", "content": QUESTION},     
        ]
    

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= mes,
        temperature = temp
    )

    return response['choices'][0]['message']['content']


def generate_answer_s(QUESTION, information, model):
    prompt = "Привет! Ты ассистент помошник по использованию програмного продукта 'Дозор' \
            'Дозор'. Будь вежлив и краток. Здоровайся с каждым пользователем! Используй эту информайию:" + information +\
            " чтобы отвветить на мой вопрос: " + QUESTION

    text = openai.Completion.create(
    model=model,
    prompt= prompt,
    max_tokens=1000,
    temperature=0
    )

    answer = text['choices'][0].text
    return answer


app = Flask(__name__)
app.config.from_object(Config)
openai.api_key = "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"
bootstrap = Bootstrap(app)

data = pd.read_csv('DATAMAIN.csv')
data = data.drop('Unnamed: 0', axis=1)
data['embeddings'] = data.embeddings.apply(emb_to_list) #делает эмбеддинги нормальными в пандас
data['embeddings_tl_dr'] = data.embeddings_tl_dr.apply(emb_to_list) #делает эмбеддинги нормальными в пандас
data['embeddings_qa'] = data.embeddings_qa.apply(emb_to_list) #делает эмбеддинги нормальными в пандас)


@app.route('/')
@app.route('/index')
def index():
    model_name = 'gpt-3'
    answer =  'Ответ на вопрос!'
    questions = "Вопрос"
    titles = ["вариант",'вариант 2']
    distance = [99,100]
   
    content = {'questions':questions, 
               'answer_content':answer, 
               'context': titles, 
               'len_context':len(titles),
                'distance':distance,
                'model_name':model_name}

    return render_template("bot.html", **content)


@app.route("/chat/", methods=["POST"])
def move_forward():
    model_name = 'gpt-3.5-turbo'
    questions =  request.form.get('question')
    context, titles, distance  =create_context_3ver(questions, data)
    answer  =  get_answer_from_gpt(questions, context, model_name)

    content = {'questions':questions, 
               'answer_content':answer, 
               'context': titles, 
               'len_context':len(titles),
                'distance':distance,
                'model_name': model_name
                }

    return render_template("bot.html", **content)








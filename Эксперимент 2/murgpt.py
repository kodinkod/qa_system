import docx 
import os
from scipy.spatial.distance import cosine
from openai.embeddings_utils import distances_from_embeddings
import openai
import warnings

openai.api_key = "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"
get_counts_token = lambda encoding, str: len(encoding.encode(str)) # коллисество токенов в тексте


def get_list_files(path, extensions):
    """
    Получаем список файлов с расширениями из extensions по пути patch
    Args:
        path (str):путь к директории
        extensions (list): массив расширений ['.txt', '.docx' ... ]
    Returns:
        list: пути подходящих файлов
    """
    names_file = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            split_tup = os.path.splitext(file)
            if split_tup[1] in extensions:
                names_file.append(os.path.join(path, file))
    return names_file



def read_docs(docs_patch):
    """Получаем информацию по заголовкам из документа
    docs_patch (list): список путей к файлам

    Returns:
        Map: словарь - заголовок->объект 
    """
    info = {}
    struct = {}
    info['start'] = "" # для шапок и вводных частей документа

    for patch in docs_patch:
        doc = docx.Document(patch)
        f = False

        
        head_title = ""
        for paragraph in doc.paragraphs:
            
            if (paragraph.style.name == 'Heading 2'):
                f = True                                                    # метка - встретили уже один headline 2
                next_title_text = head_title + " " + paragraph.text
                info[next_title_text] = ""
 
                if head_title in struct:
                    struct[head_title].append(paragraph.text)               # добавляем в структуру заголовок
                else:
                    struct[next_title_text] = [paragraph.text]


            elif (paragraph.style.name == 'Heading 1'):
                head_title = paragraph.text     
                info[head_title] = ""
                struct[head_title] = []

            elif f:
                info[next_title_text] += paragraph.text

            else:
                struct['start'] = []
                info['start'] += paragraph.text

    return {'content': info,  'struct': struct}






def create_context_2ver(question, df, max_len=1800, cont_name = 'content', emb_name = 'embeddings', count_name = 'count'):
    """поиск и генерация контекста для вопроса 

    Args:
        question (str): Текст вопроса
        df (pd.DataFrame): База данных с информацией
        max_len (int): Максимальная длина подсказки Defaults to 1800.
        cont_name (str):  название колонки с контентом 
        emb_name (str, optional): Название столбца с ембедингами Defaults to 'embeddings'.
        count_name (str, optional): Название столбца с колличеством токенов элементов promt Defaults to 'count'.

    Returns:
        text: текст подсказки для модели, который поможет ответить на вопрос
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df[emb_name].values, distance_metric='cosine')

    returns = []
    cur_len = 0
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        cur_len += row[count_name] + 4 # прибавляем токены
        
        #If the context is too long, break
        if cur_len > max_len:
            break
        
        returns.append(row[cont_name])

    # Return the context
    return "\n\n###\n\n".join(returns)



def create_context_1ver(question, df, max_len=1800, size="ada"):
    """
    Поиск контекста
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Этот алгоритм потом можно улучшить, но а пока мы просто смотрим на топ 1
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        #cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        #if cur_len > max_len:
        #    break
        
        returns.append(row["content"])
        break

    # Return the context
    return "\n\n###\n\n".join(returns)


def create_context_3ver(question, df, max_len=1800, cont_name = 'content', emb_name = 'embeddings', count_name = 'count', threshold = 0.3):
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
    if threshold < 0.17:
        warnings.warn("Слишком мальенький threshold.  Рекомендуемое значение threshold > 0.17")

    help_finde = False #помощь не нашли в данных 

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df[emb_name].values, distance_metric='cosine')

    returns = []
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

    # Return the context
    return "\n\n###\n\n".join(returns)



def get_answer(data):
    """Получение отета от gpt_3.5

    Args:
        data: данные с api gpt_3.5

    Returns:
        str:  строка вида роль: ответ
    """
    text = data['choices'][0]['message']['content']
    role = data['choices'][0]['message']['role']
    total_tokens = data["usage"]['total_tokens']

    return role + ': ' + text 



def generate_answer(QUESTION, information, mes_add=[], temp = 0, curr_count = 0):
    MAX_CONTEXT = 3
    
    mes = [
            {"role": "system", "content": "Привет! Ты ассистент помошник по использованию програмного продукта 'Дозор' \
            'Дозор'. Будь вежлив и краток. Здоровайся с каждым пользователем!"},
            {"role": "user", "content": "Главное не отвечай ничего лишнего. Говори только о том, что знаешь."},
            {"role": "user", "content": "Используй эту информацию для обоснования ответа на мой вопрос: " +information},
            {"role": "user", "content": QUESTION},     
        ]
    
    if mes_add and (curr_count < MAX_CONTEXT):
        #если мы добавляем контекст, то добавляем его 
        mes = [
            {"role": "system", "content": "Ты ассистент помошник по использованию програмного продукта 'Дозор'"},   
        ]
        
        for item in mes_add:
            mes.append(item)

        mes.append({"role": "user", "content": "Используй эту информацию для обоснования ответа на мой вопрос: " +information})
        mes.append({"role": "user", "content": QUESTION})
        
        print(100*'-')
        print(mes)
        print(100*'-')

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= mes,
        temperature = temp
    )
    answer = get_answer(response)
    return answer



def emb_to_list(text):
    parse = text.strip('[]').replace(' ', '').split(',')
    return list(map(float, parse))


def make_TL_DR():

    short_text = ""

    return short_text


def generate_answer_from_promt(mes):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= mes
    )
    answer = get_answer(response)
    return answer
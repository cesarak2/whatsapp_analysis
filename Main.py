# Author Cesar Krischer
# 01/01/2023
# Main analysis of a chat between a group of friends on Whatsapp

import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
import os
import nltk
nltk.download('punkt')
nltk.download('popular')
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams

def get_data():
    """Return content from the chat history file"""
    with open('history_23.01.03_Cond.txt', 'rt', encoding="utf-8") as file:
        content = file.read()
    return content


def split_raw_data(raw_data):
    """
    Splits data 
    """
    pass

# = = = = = = = = = = = = = = = = = = = = = = = = 
#              PARSING RAW DATA                 #
# = = = = = = = = = = = = = = = = = = = = = = = = 
# reads in text and splits around the time
data = get_data()
data = data.replace('\u200e', '').replace('\u200d', '') # removes non-printable Special. Char.
raw_chat_text = re.split('\\n\[(\d+\/\d+\/\d+,\s\d{2}:\d{2}:\d{2})\] ', data) # SPECIAL MESSAGES DON'T HAVE ':', SO ONE CAN'T USE THEM TO PARSE

chat_history_df = pd.DataFrame(raw_chat_text[2:-1]) #starts at the third message for re splitting purposes
# splits raw data into df datetime and another df for person and message (special messages follow a different structure)  
datetime_array = chat_history_df.iloc[1::2].reset_index(drop = True)
person_and_messages_array = chat_history_df.iloc[::2].reset_index(drop = True) # name and message are separated by a ': '
person_and_messages_array[['name', 'message']] = person_and_messages_array[0].str.split(': ', expand=True, n=1)

result = pd.concat([datetime_array, person_and_messages_array], axis=1)
result.columns = ['datetime', 'raw', 'person', 'message']

# KEEP ONLY SPECIAL MESSAGES (MESSAGE IS NULL)
special_messages_df = result.loc[result['message'].isna()]
special_messages_df.columns = ['datetime', 'raw', 'message', 'blank']
special_messages_df = special_messages_df[['datetime', 'message']] #kepps only person

# CHAT WITHOUT SPECIAL MESSAGES
chat = result.loc[result['message'].notna()]
#chat.columns = ['datetime', 'raw', 'person', 'message']
chat = chat[['datetime', 'person', 'message']] #kepps only person
chat['datetime'] = pd.to_datetime(chat['datetime'])
chat['message_length'] = chat['message'].str.len()
chat.drop(chat[chat['person'] == 'You'].index, inplace=True) # videocalls are made by 'you'

# = = = = = = = = = = = = = = = = = = = = = = = = 
#         MANUAL VARIABLES CALCULATION          #
# = = = = = = = = = = = = = = = = = = = = = = = = 
group_users = chat.person.unique()
if ['You'] in group_users:
    print(len(group_users)-1) 
else:
    print(len(group_users)) # who exported the chat never videoconferenced the group

for user in group_users:
    print(user + ', ' + str(len(chat[chat['person'] == user])))


# = = = = = = = = = = = = = = = = = = = = = = = = 
#                  EXPLORATORY                  #
# = = = = = = = = = = = = = = = = = = = = = = = = 
messages_per_person = chat.groupby('person').count().\
                           drop('datetime', axis=1).\
                            sort_values(by='message', ascending=False)
#messages_per_person = messages_per_person.drop('You', axis = 0)
messages_per_person.drop('message', axis=1)
people = messages_per_person.index

messages_per_person['total_characters'] = chat.groupby('person').sum(numeric_only=True)
messages_per_person['char_per_message_avg'] = chat.groupby('person').mean(numeric_only=True)
messages_per_person['char_per_message_median'] = chat.groupby('person').median(numeric_only=True)
messages_per_person['char_per_message_std'] = chat.groupby('person').std(numeric_only=True)
messages_per_person['min_len_message'] = chat.groupby('person').min(numeric_only=True)
messages_per_person['max_len_message'] = chat.groupby('person').max(numeric_only=True)
messages_per_person.sort_values(by='char_per_message_avg', ascending=False)



chat.groupby('person').hist()

#chat['message_length'].hist(by=chat['person'])
# = = = = = = = = = = = = = = = = = = = = = = = = 
#                    BIGRAMS                    #
# = = = = = = = = = = = = = = = = = = = = = = = = 

def get_ngrams(df, text_column, bigram_size=2, language='portuguese'):
    '''
    given a df, removes stop words for a given language and returns the bigrams.
    inputs: dataframe, the column containing the raw text, the bigram size, the stop words language. 
    outputs: a counter with the words and their counts.
    '''
    stop_words = set(stopwords.words('portuguese'))
    if language == 'portuguese':
        aditional_stop_words = ['this', 'deleted', 'message', 'was', 'vc', 'vcs',
                        'vote', 'votes', 'q', 'é', 'pra', 'image',
                        'omitted', 'audio', 'sticker', 'gif', 'deleted', '•',
                        'nao', 'pq', 'eh', 'tá', 'pra', 'to']
    stop_words.update(aditional_stop_words)
    df['message_no_stop_words'] = df[text_column].apply(lambda x: ' '.join([word.lower() for word in x.lower().split() if word not in (stop_words)]))
    df['most_used_words'] = df['message_no_stop_words'].apply(lambda row: list(nltk.ngrams(row.split(' '),bigram_size)))
    total_bigrams = df['most_used_words'].to_numpy()
    bigram_flat_list = [item for sublist in total_bigrams for item in sublist]
    counted_bigrams = Counter(bigram_flat_list)
    return counted_bigrams.most_common(20)

get_ngrams(df = chat, text_column = 'message', bigram_size = 2, language = 'portuguese')

'''
# CONVERTED TO A FUNCTION ABOVE, TO BE REMOVE ON NEXT COMMIT
stop_words = set(stopwords.words('portuguese'))
aditional_stop_words = ['this', 'deleted', 'message', 'was', 'vc', 'vcs',
                        'vote', 'votes', 'q', 'é', 'pra', 'image',
                        'omitted', 'audio', 'sticker', 'gif', 'deleted', '•',
                        'nao', 'pq', 'eh', 'tá', 'pra', 'to']
stop_words.update(aditional_stop_words)

#chat['tokenized'] = chat['message'].apply(lambda row: row.lower().split())
#chat['message_no_stop_words'] = 0
chat['message_no_stop_words'] = chat['message'].apply(lambda x: ' '.join([word.lower() for word in x.lower().split() if word not in (stop_words)]))


#chat['message_no_stop_words'][180:182]
#chat['message'][180:190]


chat['most_used_words'] = chat['message_no_stop_words'].apply(lambda row: list(nltk.ngrams(row.split(' '),1)))
chat['bigrams'] = chat['message_no_stop_words'].apply(lambda row: list(nltk.bigrams(row.split(' '))))
chat['trigrams'] = chat['message_no_stop_words'].apply(lambda row: list(nltk.trigrams(row.split(' '))))
chat['4grams'] = chat['message_no_stop_words'].apply(lambda row: list(nltk.ngrams(row.split(' '),4)))
chat['5grams'] = chat['message_no_stop_words'].apply(lambda row: list(nltk.ngrams(row.split(' '),5)))

# BIGRAMS PER PERSON
total_bigrams = chat['most_used_words'].to_numpy()
total_bigrams = chat['bigrams'].to_numpy()
total_bigrams = chat['trigrams'].to_numpy()
total_bigrams = chat['4grams'].to_numpy()
total_bigrams = chat['5grams'].to_numpy()

#total_bigrams[55]
bigram_flat_list = [item for sublist in total_bigrams for item in sublist]
counted_bigrams = Counter(bigram_flat_list)
counted_bigrams.most_common(20)
#sorted(counted_bigrams.items(), key=lambda pair: pair[1], reverse = True)
'''



# = = = = = = = = = = = = = = = = = = = = = = = = 
#                CHAT ENGAGEMENT                #
# = = = = = = = = = = = = = = = = = = = = = = = = 
# https://plotly.com/python/network-graphs/



# = = = = = = = = = = = = = = = = = = = = = = = = 
#                     FFT                       #
# = = = = = = = = = = = = = = = = = = = = = = = = 



# = = = = = = = = = = = = = = = = = = = = = = = = 
#                    CHARTS                     #
# = = = = = = = = = = = = = = = = = = = = = = = = 
# SIZE MESSAGE
x = messages_per_person.message_length
#y = messages_per_person.char_per_message_avg
y = messages_per_person.char_per_message_median

fig, ax = plt.subplots()
ax.scatter(x, y)
plt.xlabel('Total number of messages sent')
plt.ylabel('Characters per message, median')

for i, txt in enumerate(messages_per_person.index):
    ax.annotate(txt, (x[i], y[i]))


# CHARTS PER PERSON
plt.rcdefaults()
fig, ax = plt.subplots()

y_pos = np.arange(len(people))

ax.barh(y_pos, messages_per_person['message'], align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('number of messages')
ax.set_title('Who participates more in this group?')

plt.show()
# = = = = =
chat.groupby([chat['datetime'].dt.date]).mean()

plt.plot(chat.datetime, linestyle = 'dotted')
plt.show()
# = = = = 
df_month = chat.resample('M', on='datetime').count()
df_day = chat.groupby(pd.Grouper(key='datetime', axis=0, freq='D')).count()
df_day = chat.groupby(pd.Grouper(key='datetime', axis=0, freq='W')).count()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df_month.index, df_month.message);

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df_day.index, df_day.message);



# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = messages_per_person.index
sizes = messages_per_person['message']
explode = (0, 0.1, 0, 0)


fig1, ax1 = plt.subplots()
ax1.pie(sizes, autopct='%1.1f%%', labels=labels,
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
# = = = = = =



# = = = = = = = = = = = = = = = = = = = = = = = = 
#                SPECIAL MESSAGES               #
# = = = = = = = = = = = = = = = = = = = = = = = = 
# special phrases:
reserved_phrases = ['Messages and calls are end-to-end encrypted. \
                     No one outside of this chat, not even WhatsApp, \
                     can read or listen to them.',
                    'created group',
                    'added',
                    "changed this group's icon",
                    'changed the group description',
                    'changed their phone number'] 


special_messages_df = raw_chat.loc[raw_chat['message'].isna()]
special_messages_df = special_messages_df[['datetime', 'person']] #kepps only person
special_messages_df.columns = ['datetime', 'special_message']

len(raw_chat['person'].astype('category'))

def label_special_phrase(row):
    if row['special_message'] == 'João Bonn left':
        return 'po po po'

special_messages_df.apply(lambda row: label_special_phrase(row), axis = 1)



# = = = = = = = = = = = = = = = = = = = = = = = = 
#                   REFERENCES                  #
# = = = = = = = = = = = = = = = = = = = = = = = = 
# Bigrams:
# https://stackoverflow.com/questions/54694038/forming-bigrams-of-words-in-a-pandas-dataframe
# Flatten lists
# https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
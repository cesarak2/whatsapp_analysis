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
import plotly.express as px
import scipy.fft

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
#                    N-GRAMS                    #
# = = = = = = = = = = = = = = = = = = = = = = = = 

def get_ngrams(df, text_column, bigram_size=2, language='portuguese'):
    '''
    given a df, removes stop words for a given language and returns the bigrams.
    inputs: dataframe, the column containing the raw text, the bigram size, the stop words language. 
    outputs: a counter with the words and their counts.
    '''
    stop_words = set(stopwords.words(language))
    stop_words.update(['this', 'deleted', 'message', 'was', '•', 
                       'omitted', 'audio', 'sticker', 'gif', 'deleted',
                       'vote)', 'votes)', 'to']) # words used by Whatsapp regarding media sent
    if language == 'portuguese': # the main group analysis was made on a pt speaking chat
        aditional_stop_words = ['vc', 'vcs', 'vote', 'votes', 'q',
                                'é', 'pra', 'image', 'n',
                                'nao', 'pq', 'eh', 'tá', 'pra']
        stop_words.update(aditional_stop_words)
    df['message_no_stop_words'] = df[text_column].apply(lambda x: ' '.join([word.lower() for word in x.lower().split() if word not in (stop_words)]))
    df['most_used_words'] = df['message_no_stop_words'].apply(lambda row: list(nltk.ngrams(row.split(' '),bigram_size)))
    total_bigrams = df['most_used_words'].to_numpy()
    bigram_flat_list = [item for sublist in total_bigrams for item in sublist]
    counted_bigrams = Counter(bigram_flat_list)
    return counted_bigrams.most_common(20)

get_ngrams(df = chat, text_column = 'message', bigram_size = 1, language = 'portuguese')
get_ngrams(df = chat, text_column = 'message', bigram_size = 2, language = 'portuguese')
get_ngrams(df = chat, text_column = 'message', bigram_size = 3, language = 'portuguese')

get_ngrams(df = chat, text_column = 'message', bigram_size = 1, language = 'english')

# = = = = = = = = = = = = = = = = = = = = = = = = 
#                CHAT ENGAGEMENT                #
# = = = = = = = = = = = = = = = = = = = = = = = = 
# https://plotly.com/python/network-graphs/

# Each timestamp will 

# SUPPOSITIONS:
#   – people that ta
#   – chat is definied by 

group_users
len(group_users)


def score_by_time(x, a=0.42, b=30):
    '''
    calculates y = (a*e)^(b-x)
    Used to calculate the score for a given delta time
    '''
    return (a*np.e)**-(x-b)

def time_by_score(y, a=0.42, b=30):
    '''
    Solves y = (a*e)^(b-x) for x
    Used to calculate what is the delta time for a given score
    Std values have appr.: initial value=50, half life=5min, 30min to reach score=1
    '''
    if y <= 0 or y > 1E19: # in case the delta is miscalculated
        return 0
    result = (b+b*np.log(a)-np.log(y))/(1+np.log(a))
    
    if type(result) is np.float64:
        return result
    else: # just a big number
        return 4E2 # just a big number

#time_by_score(50)
#score_by_time(-4.756150318632312)

score_by_time(0.5)
time_by_score(26.62460500158606)

53.24921000317212/2

# creates deltatime in min (converted from s for precision) to be used to calculate score
points_conversation_df = chat[['datetime', 'person']] # new df is created, not copied
points_conversation_df['delta_time'] = points_conversation_df['datetime'].diff().\
                                                    astype('timedelta64[s]')/60
#score_by_time(points_conversation_df.delta_time.iloc[10]) #testing calculating score
#points_conversation_df['Giordano Jobim'] = np.where(points_conversation_df['person'] == 'Giordano Jobim', score_by_time(0),'')
#points_conversation_df.loc[:,'Giordano Jobim'] = np.where(points_conversation_df['person'] == 'Giordano Jobim', 30,'')

# draft to create columns
# create a column per person, and if that row represents that person talking, assign a high score, otherwise a low score.
for person in group_users:
    points_conversation_df.loc[:,person] = np.where(points_conversation_df['person'] == person, score_by_time(0),0.1)

# the score decreases with time by an exponential factor, based on last time that person spoke
# calculates the score by deltatime for all rows. 
points_conversation_df.apply(lambda row: score_by_time(row['delta_time']) if True else 0, axis=1)


def get_index_of_first_message(target_person, df):
    '''
    returns an int with the first message index a certain person sent
    inputs: a person's name to search in the df and the df containing the column named "person"
    output: index as int
    '''
    try: # finds the first occurance of a message send by that person
        first_index_person = df[df.person == target_person].index[0] 
    except IndexError:
        first_index_person = 0 #if the person never had spoken, return 0
    return first_index_person



for current_person in group_users:
    index_first_message = get_index_of_first_message(target_person=current_person, df=points_conversation_df)
    print(f'current_person={current_person}, index_first_message={index_first_message}')
    if index_first_message == 0: # if the person never sent a message, ignore them
        break
    for i in range(index_first_message, len(points_conversation_df)): # from the person's 1st message until the end
        if points_conversation_df.person.iloc[i] == current_person:  
            pass # high score when person speaks was already defined
        else: # a different person is talking
                # converts the score to time, adds that to the time delta to the last message and converts back to score
            points_conversation_df[current_person].iloc[i] = score_by_time( #transforms back in score
                                                                           time_by_score(points_conversation_df[current_person].iloc[i-1])\
                                                                           + points_conversation_df.delta_time.iloc[i]) # both are times
# points_conversation_df



# matrix is the groupby person
score_heatmap_df = points_conversation_df.groupby('person').sum().drop('delta_time', axis=1)
score_heatmap_df = score_heatmap_df.reindex(sorted(score_heatmap_df.columns), axis=1) #reorder alphabetically

for i in range(0,len(score_heatmap_df)): #because the rows order match the columns order
    score_heatmap_df.iloc[i,i] = '' #erase the person self score

#plotly chart
fig = px.imshow(score_heatmap_df,
                labels=dict(x="Engager person", y="Speaker person", color="Total Score"))
fig.update_xaxes(side="top", tickangle=55)
fig.show()
# # # # # 


#points_conversation_df.to_csv('points_conversation_df.csv')
#help(get_index_of_first_message
plt.plot(points_conversation_df.delta_time)

# = = = = = = = = = = = = = = = = = = = = = = = = 
#                     FFT                       #
# = = = = = = = = = = = = = = = = = = = = = = = = 
np.array(points_conversation_df.delta_time[5:])
delta_time_array = np.array(points_conversation_df.delta_time[5:])# / np.timedelta64(1,'s'))
fft = scipy.fft.fft(delta_time_array)
plt.plot(np.abs(fft))
plt.ylim(0,300000)
plt.title('FFT per timedelta')
plt.show() # no valid frequency exists, hence this method can't be applied.

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
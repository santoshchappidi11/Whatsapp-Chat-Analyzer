import matplotlib.pyplot as plt
import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import emoji
import pandas as pd
# import seaborn as sns
# import plotly.express as px
# from collections.abc import Iterable
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from sklearn.feature_extraction.text import CountVectorizer
import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# cv = TfidfVectorizer()

with open('svm_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)


# with open('svm_model.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)



st.set_page_config(page_title="santosh", layout="wide")
st.title("WhatsChatAnalyze🔎")
st.write("*Made by Santosh!👨🏻‍💻*")
st.sidebar.title("WhatsApp Chat Analyzer with Sentiment Analysis")
uploadedFile = st.sidebar.file_uploader("Choose a File🗃️")
if uploadedFile is not None:
    bytesData = uploadedFile.getvalue()
    finalData = bytesData.decode("utf-8")
    dataFrame = preprocessor.preprocess(finalData)
    #st.dataframe(dataFrame.head())

    # fetch unique users
    userList = dataFrame["user"].unique().tolist()
    if ("default" in userList):
        userList.remove("default")
    userList.sort()
    userList.insert(0, "Overall")
    selectedUser = st.sidebar.selectbox("Show Analysis🤔 WRT", userList)


    if (True):
        # statistics-------------------------------------------------------------->
        numMessages, numWords, numMedia, numURL = helper.fetchStats(
            selectedUser, dataFrame)
        st.title("Top Statistics📈")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages🤳🏻")
            st.title(numMessages)
        with col2:
            st.header("Total Words💭")
            st.title(numWords)
        with col3:
            st.header("Media Shared🎥")
            st.title(numMedia)
        with col4:
            st.header("Links Shared🔗")
            st.title(numURL)

        # monthly timeline----------------------------------------------------------->
        st.title("Monthly Timeline⌚")
        timeline = helper.monthlyTimeline(selectedUser, dataFrame)
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 3))
        plt.plot(timeline['time'], timeline['message'])
        plt.xticks(rotation='vertical')
        plt.title(f"{selectedUser}", color='yellow')
        plt.xlabel('Month', color='white')
        plt.ylabel('Message Count', color='white')
        st.pyplot(plt)

        # daily timeline----------------------------------------------------------->
        st.title("Daily Timeline📅")
        dailyTimeline = helper.dailyTimeline(selectedUser, dataFrame)
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 3))
        plt.plot(dailyTimeline['onlyDate'], dailyTimeline['message'])
        plt.xticks(rotation='vertical')
        plt.title('Daily Message Count', color='yellow')
        plt.xlabel('Date', color='white')
        plt.ylabel('Message Count', color='white')
        st.pyplot(plt)

        # activity map----------------------------------------------------------->
        st.title("Week Activity📊")
        col1, col2 = st.columns(2)
        weekActivitySeries, weekActivity = helper.weekActivity(selectedUser, dataFrame)
        weekActivity = weekActivity.sort_values('message')
        days = weekActivity['dayName']
        messages = weekActivity['message']
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(messages, labels=days, autopct='%1.1f%%', colors=plt.cm.Dark2.colors)
            ax.axis('equal')
            plt.style.use('dark_background')
            st.pyplot(fig)
            
            
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(days, messages)
            ax.set_xlabel('Day of the Week', color="yellow")
            ax.set_ylabel('Number of Messages', color='yellow')
            plt.style.use('dark_background')
            st.pyplot(fig)
        
        #Month Activity--------------------------------------------------------------------->
        st.title("Month Activity📊")
        col1, col2 = st.columns(2)
        monthActivitySeries, monthActivity = helper.monthActivity(selectedUser, dataFrame)
        monthActivity = monthActivity.sort_values('message')
        month = monthActivity['monthName']
        messages = monthActivity['message']
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(messages, labels=month, autopct='%1.1f%%', colors=plt.cm.Dark2.colors)
            ax.axis('equal')
            plt.style.use('dark_background')
            st.pyplot(fig)
            
            
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(month, messages)
            ax.set_xlabel('Month of the Year', color="yellow")
            ax.set_ylabel('Number of Messages', color='yellow')
            plt.style.use('dark_background')
            st.pyplot(fig)
            
        #hourly activity----------------------------------------------------------->
        st.title("Hour Activity⌛")
        h1, h2 = helper.hourActivity(selectedUser, dataFrame)
        
        fig, ax = plt.subplots(figsize=(12, 3))
        h1.unstack('dayName').plot(ax=ax)
        ax.set_xlabel('Hour of the Day', color='yellow')
        ax.set_ylabel('Number of Messages', color='yellow')
        ax.set_title('Messages Sent by Hour of the Day', color='white')
        plt.style.use('dark_background')
        st.pyplot(fig)
        
                
        #period activity----------------------------------------------------------->
        st.header("Activity by Time Period📲")
        activity = helper.activity(selectedUser, dataFrame)
        activity = activity.sort_values('message')
        period = activity['period']
        messages = activity['message']
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.bar(period, messages)
        ax.set_xlabel('Period', color="yellow")
        ax.set_ylabel('Number of Messages', color='yellow')
        ax.set_title('Activity Chart')
        plt.style.use('dark_background')
        st.pyplot(fig)


        # finding busiest users in the group----------------------------------------------------------->
        if selectedUser == 'Overall':
            st.header("Top Chatters🗣️")
            topChatter, topChatterPercent = helper.mostBusy(dataFrame)
            col1, col2 = st.columns(2)

            with col1:
                plt.style.use('dark_background')
                name = topChatter.index
                name = [emoji.emojize(n) for n in name]
                count = topChatter.values
                fig, ax = plt.subplots()
                plt.xlabel('Name').set_color('yellow')
                plt.ylabel('Messages Sent').set_color('yellow')
                ax.bar(name, count, width=0.8)
                plt.xticks(rotation='vertical')
                ax.tick_params(axis='both', which='major', labelsize=8)

                st.pyplot(fig)

            with col2:
                st.dataframe(topChatterPercent)
                
        
           
        # most common words----------------------------------------------------------->
        mostCommon = helper.mostCommon(selectedUser, dataFrame)
        if (mostCommon.shape[0] != 0):
            st.header("Top Words Used🥇")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                plt.ylabel('Message').set_color('yellow')
                plt.xlabel('Frequency').set_color('yellow')
                ax.barh(mostCommon['Message'], mostCommon['Frequency'])
                plt.xticks(rotation="vertical")
                st.pyplot(fig)

            with col2:
                st.dataframe(mostCommon)


        # emoji analysis----------------------------------------------------------->
        emoji_df = helper.mostEmoji(selectedUser, dataFrame)
        if (emoji_df.shape[0] != 0):
            st.title("Emoji Analysis😳")
            st.dataframe(emoji_df)
             

        # Perform sentiment analysis----------------------------------------------------------->
        # Define a mapping between sentiment labels and emojis
        st.title("Sentiment Analysis 😊-😐-😔")
        emoji_mapping = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😔'
        }

        # Assuming model and text are defined
        texts = dataFrame['message']

        user = dataFrame['user']

        # Fit CountVectorizer with the vocabulary from training data
        # vectorizer.fit(texts)

        # Now transform the text data
        text_cv = vectorizer.transform(texts)
        predictions = [model.predict(text_cv[i])[0] for i in range(len(texts))]

        # Map sentiment predictions to emojis
        emojis = [emoji_mapping[sentiment] for sentiment in predictions]

        # Create a DataFrame to store the messages and their corresponding predictions
        df = pd.DataFrame({ 'User':user,  'Message': texts, 'Prediction': predictions, 'Emoji': emojis})

        # Display DataFrame in tabular format using Streamlit
        st.write(df)


# ------------------------------------------------------------------------------------------->
        # def analyze_sentiment(text):
        #     sid = SentimentIntensityAnalyzer()
        #     scores = sid.polarity_scores(text)
        #     # Classify sentiment based on compound score
        #     if scores['compound'] >= 0.05:
        #         return 'positive'
        #     elif scores['compound'] <= -0.05:
        #         return 'negative'
        #     else:
        #         return 'neutral'

        # # Function to add emojis to sentiments
        # def add_emojis(sentiment):
        #     emoji_mapping = {
        #         'positive': '😊',
        #         'neutral': '😐',
        #         'negative': '😔'
        #     }
        #     return emoji_mapping.get(sentiment, '')

        # # Main function
        # def main(dataFrame):
        
        #     # Assuming dataFrame is your DataFrame containing the messages
        #     messages = dataFrame['message']
        #     users = dataFrame['user']
            
        #     # Create empty lists to store messages and sentiments
        #     messages_list = []
        #     users_list = []
        #     sentiments_list = []
        #     emojis_list = []
            
        #     # Analyze sentiment for each message
        #     for message, user in zip(messages, users):
        #         sentiment = analyze_sentiment(message)
        #         messages_list.append(message)
        #         users_list.append(user)
        #         sentiments_list.append(sentiment)
        #         emojis_list.append(add_emojis(sentiment))
            
        #     # Create DataFrame from the lists
        #     df = pd.DataFrame({'User': users_list, 'Message': messages_list, 'Sentiment': sentiments_list})

        #     # Add emojis to the Sentiment column
        #     df['Emoji'] = df['Sentiment'].apply(add_emojis)
            
            
        #     # Display DataFrame in tabular format
        #     st.write(df)

        # if __name__ == "__main__":
        #     main(dataFrame)

        
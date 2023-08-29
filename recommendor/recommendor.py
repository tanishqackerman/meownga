import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def update_dataset(csv_file, json_data):
    df = pd.read_csv(csv_file)
    json_df = pd.DataFrame(json_data, index=[0])
    df = df.append(json_df, ignore_index=True)
    df.to_csv(csv_file, index=False)


def recommend(csv_file, manga_id):
    df = pd.read_csv(csv_file)
    df = df[["mal_id", "title", "genres", "author", "synopsis"]]

    def convert_list_to_string(s):
        genre_list = s[1:-1].split("', '")
        joined_genre = ' '.join(genre_list)
        return joined_genre[1: -1]

    df["genres"] = df["genres"].apply(convert_list_to_string)
    df["author"] = df["author"].apply(convert_list_to_string)
    df["features"] = ' '.join(df.iloc[0].astype(str))
    newdf = df[["mal_id", "title", "features"]]
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(newdf["features"]).toarray()
    similarity = cosine_similarity(vectors)
    manga_index = newdf[newdf["id"] == manga_id].index[0]
    distances = similarity[manga_index]
    manga_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1: 6]
    for i in manga_list:
        print(newdf.iloc[i[0]].title)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

app = Flask(__name__)


def update_dataset(json_data):
    df = pd.read_csv("manga_dataset.csv")
    json_df = pd.DataFrame(json_data, index=[0])
    df = df.append(json_df, ignore_index=True)
    df.to_csv("manga_dataset.csv", index=False)


def recommend(manga_id):
    df = pd.read_csv("manga_dataset.csv")
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
    return manga_list


@app.route('/recommendor', methods=['POST'])
def recommendor():
    try:
        data = request.json
        on_action = data.get('on_action')

        if on_action == 'recommend':
            mangaid = data.get('mangaid')
            if mangaid is None:
                return jsonify({'error': 'mangaid is required for recommendation'}), 400

            manga_list = recommend(mangaid)
            return jsonify({'recommendation': manga_list})

        elif on_action == 'update':
            update_json = data.get('manga')
            if update_json is None:
                return jsonify({'error': 'manga is required for update action'}), 400

            update_dataset(update_json)
            return jsonify({'update_result': "success"})

        else:
            return jsonify({'error': 'Invalid on_action value'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

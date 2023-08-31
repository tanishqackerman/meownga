from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)


def sentimentAnalysis(comment):
    classifier = pipeline("sentiment-analysis")
    sentiment = classifier(comment)
    return sentiment[0]['label']


@app.route('/commentSentiment', methods=['POST'])
def comment_sentiment():
    try:
        data = request.get_json()
        comment = data.get('comment')

        if comment is None:
            return jsonify({'error': 'Comment field is missing'}), 400

        processed_comment = sentimentAnalysis(comment)
        return jsonify(
            {'comment': comment, 'sentiment': processed_comment})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from flask import Flask, request, jsonify

app = Flask(__name__)


# for step in range(4):
#     new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
#
#     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
#
#     chat_history_ids = model.generate(
#         bot_input_ids, max_length=200,
#         pad_token_id=tokenizer.eos_token_id,
#         no_repeat_ngram_size=3,
#         do_sample=True,
#         top_k=100,
#         top_p=0.7,
#         temperature=0.8
#     )
#
#     print("JoshuaBot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


def generate_reply(id, message):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model = AutoModelWithLMHead.from_pretrained(id)

    new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(
        bot_input_ids, max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


@app.route('/chatbotReply', methods=['POST'])
def chatbot_response():
    data = request.get_json()
    if 'id' in data and 'message' in data:
        id = data['id']
        message = data['message']
        reply = generate_reply(id, message)
        return jsonify({'reply': reply})
    else:
        return jsonify({'error': 'id or message not found in request'}), 400


if __name__ == '__main__':
    app.run(debug=True)

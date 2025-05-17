"""
Bot to transcribe audio messages using Hugging Face's whisper-large-v3 model.

```python
python telegram_gpt.py
```


To set commands (after running /setcommands from BotFather):
```
clear - Clear chat history.
n_words - How many words for response.
listen - Listen to the bot's response.
```

Press Ctrl-C on the command line to stop the bot.
"""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import requests
from telegram.ext import Application, ContextTypes, MessageHandler, filters, CommandHandler, CallbackQueryHandler
from keys import TELEGRAM_KEY, HUGGING_FACE_KEY, OPENAI_KEY, TOGETHER_AI
from pprint import pprint
# from openai import OpenAI
# import torch
from together import Together

from PIL import Image
import numpy as np
import base64

# For image retrieval
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

MAX_CHAT_HISTORY = 10
VERBOSE = True
LOCAL_ASR = False
# CUDA_AVAILABLE = torch.cuda.is_available()
USE_TOGETHER_AI = True
TELEGRAM_MAX_OUTPUT = 4096    # Telegram max output length


global USER_MESSAGES    # user message history
global N_WORDS    # number of words for response

USER_MESSAGES = dict()    # dict of chat/group IDs and their messages
N_WORDS = dict()

# config for speech transcription
headers = {"Authorization": f"Bearer {HUGGING_FACE_KEY}", "Content-Type": "audio/wav"}

# prepare TTS
client_tts = Together(
    api_key=TOGETHER_AI,
) 

# prepare LLM
if USE_TOGETHER_AI:
    print("Using Together AI")
    llm_model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    # llm_model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
    # llm_model = "meta-llama/Llama-Vision-Free"
    client = Together(
        api_key=TOGETHER_AI,
    ) 
else:
    print("Using OpenAI")
    """llm_model = "gpt-3.5-turbo"
    client = OpenAI(api_key=OPENAI_KEY)
"""
# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# prepare ASR
if LOCAL_ASR:
    from transformers import pipeline
    import audiofile
    import librosa

    if CUDA_AVAILABLE:
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0)
    else:
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    asr_rate = 16000

# Load the pretrained VGG16 model for image retrieval
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

path = "images/" # path to images database (cats)

# querying ASR
def query_asr(filename):

    if LOCAL_ASR:
        # TODO: check if works

        signal, sampling_rate = audiofile.read(filename)
        if sampling_rate != asr_rate:
            signal = librosa.resample(signal, orig_sr=sampling_rate, target_sr=asr_rate)
        output = asr_pipe(signal, generate_kwargs={"language": "english"})
        return output

    else:
        # Hugging Face endpoint
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3", 
            headers=headers, 
            data=data
        )
        return response.json()


def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# querying LLM
def query_llm(input_text, user_id, image_paths=None):

    if user_id not in N_WORDS:
        N_WORDS[user_id] = -1

    if N_WORDS[user_id] > 0:
        input_text += f" (in {N_WORDS[user_id]} words or less)"

    # add to message history
    
    if image_paths:
        content = [
            {"type": "text", "text": input_text},
        ]
        for path in image_paths:
            base64_image = encode_image(path)
            content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    })
        message = {
            "role": "user",
            "content": content,
        }
    else:
        message = {"role": "user", "content": input_text}
    if user_id in USER_MESSAGES:
        USER_MESSAGES[user_id].append(message)
    else:
        USER_MESSAGES[user_id] = [message]

    # prompt LLM
    response = client.chat.completions.create(
        model=llm_model,
        messages=USER_MESSAGES[user_id]
    )
    text_response = response.choices[0].message.content
    
    # update message history
    USER_MESSAGES[user_id].append({"role": "assistant", "content": text_response})

    if VERBOSE:
        pprint(USER_MESSAGES[user_id])

    # clear chat history (if too long)
    if len(USER_MESSAGES[user_id]) > 2 * MAX_CHAT_HISTORY:
        # remove bot response
        USER_MESSAGES[user_id].pop(0)
        # remove question
        USER_MESSAGES[user_id].pop(0)

    text_response = f"MODEL: {llm_model}\n\n{text_response}"
    # truncate response if too long
    if len(text_response) > TELEGRAM_MAX_OUTPUT:
        text_response = text_response[:TELEGRAM_MAX_OUTPUT-20] + "\n\nOUTPUT TRUNCATED"
    return text_response


async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """STEP 1) Speech to text."""
    audio_file = await update.message.voice.get_file()

    # load audio into numpy array
    tmp_file = "voice_note.wav"
    await audio_file.download_to_drive(tmp_file)

    # transcription
    output = query_asr(tmp_file)
    output = output["text"]

    """STEP 2) Prompt LLM."""
    text_response = query_llm(output, update.message.from_user.id)

    # respond text through Telegram
    await update.message.reply_text(text_response)


async def text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # prompt LLM
    input_text = update.message.text
    user_id = update.message.from_user.id
    text_response = query_llm(input_text, user_id)

    # respond text through Telegram
    await update.message.reply_text(text_response)


async def n_words(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """How many words for response."""
    # button for user to accept or reject the suggestion
    keyboard = [
        [
            InlineKeyboardButton("10", callback_data="10"),  # callback_data has to be string
            InlineKeyboardButton("50", callback_data="50"),
        ],
        [
            InlineKeyboardButton("100", callback_data="100"),
            InlineKeyboardButton("None", callback_data="-1"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # get number of words
    await update.message.reply_text("Upper limit for response?", reply_markup=reply_markup)   


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""
    
    # set number of words
    query = update.callback_query
    user_id = query.from_user.id

    # update number of words
    N_WORDS[user_id] = int(query.data)

    response_text = f"Number of words for response: {N_WORDS[user_id]}"
    if VERBOSE:
        print(response_text)

    # respond text through Telegram
    await query.message.reply_text(response_text)


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear chat history."""
    user_id = update.message.from_user.id
    del USER_MESSAGES[user_id]
    await update.message.reply_text("Chat history cleared.")

async def listen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Listen to the bot's response."""
    user_id = update.message.from_user.id
    if user_id not in USER_MESSAGES:
        await update.message.reply_text("No chat history found.")
        return

    # get last message
    last_message = USER_MESSAGES[user_id][-1]["content"]

    # convert text to speech
    speech_file_path = "speech.wav"
    response = client_tts.audio.speech.create(
        model="cartesia/sonic",
        input=last_message,
        voice="1920's radioman",
    )
    response.stream_to_file(speech_file_path)

    # send audio file
    await update.message.reply_audio(speech_file_path, caption=last_message)

# Fonction pour extraire les caractéristiques d'une image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    return features.flatten()

database = {
    'Greg': extract_features(path + "Greg.jpg"),
    'Albert': extract_features(path + "Albert.jpg"),
    'Lucy': extract_features(path + "Lucy.jpg"),
    'Nala': extract_features(path + "Nala.jpg"),
    # In practice we would have more cats in the database
    # Don't name a cat "No"
}

def find_most_similar_cat(input_img_path): # Compare with cosine similarity
    input_features = extract_features(input_img_path)
    similarities = {}
    for cat_id, features in database.items():
        similarity = np.dot(input_features, features) / (np.linalg.norm(input_features) * np.linalg.norm(features))
        similarities[cat_id] = similarity
    most_similar_cat = max(similarities, key=similarities.get)
    return most_similar_cat

"""async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # 1) Is the photo a cat ?
    # a) Prompt Llama
    # b) Extract Yes or No

    # 2) If yes, determine most relevant cat, else return that it is not a cat
    # 3) Return information about the cat

    user_id = update.message.from_user.id
    photos = []
    for i, photo_data in enumerate(update.message.photo):
        photo_file = await photo_data.get_file()
        tmp_photo = f"photo_{i}.jpg"
        photos.append(tmp_photo)
        await photo_file.download_to_drive(tmp_photo)

    # load image into numpy array
    query = f"Are the cats on the following {len(photos)} images the same cat? Please be concise."
    text_response = query_llm(query, user_id, photos)

    # respond text through Telegram
    await update.message.reply_text(text_response)

    # respond photo
    # await update.message.reply_photo(tmp_photo, caption=f"Image shape: {img.shape}")"""

async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # 1) Is the photo a cat ?
    # a) Prompt Llama
    # b) Extract Yes or No

    # 2) If yes, determine most relevant cat, else return that it is not a cat
    # 3) Return information about the cat

    found_cat = "No" # No cat found yet

    user_id = update.message.from_user.id
    photos = []
    for i, photo_data in enumerate([update.message.photo[-1]]):
        photo_file = await photo_data.get_file()
        tmp_photo = f"photo_{i}.jpg"
        photos.append(tmp_photo)
        await photo_file.download_to_drive(tmp_photo)
    # print(len(photos))
    # if len(photos)>1:
    #    text_response = "Too many photos, please only submit one at the time :)"
    #else:
    # 1) Is the photo a cat ?
    # a) Prompt Llama
    query = f"Does this image represent exactly one cat ? Answer in one word : Yes or No"
    answer = query_llm(query, user_id, photos)

    # b) Extract Yes or No
    # 2) If yes, determine most relevant cat, else return that it is not a cat
    print(answer)
    if ("Yes" in answer) or ("yes" in answer) :
        answer = "yes"
        found_cat = find_most_similar_cat(photos[0])
    else:
        answer = "no"
        text_response = "I could not find a cat in this photo, please retry"
        
    # 3) Return information about the cat
    if found_cat != "No" : # If we found a cat...
        text_response = "It's " + found_cat + "!"

        query = text_response + f" Please tell me what you know about this cat, including the name. Make hypothesis about missing information about it. Be concise."
        text_response = query_llm(query, user_id, photos)

    # respond text through Telegram
    await update.message.reply_text(text_response)

    # respond photo
    # await update.message.reply_photo(tmp_photo, caption=f"Image shape: {img.shape}")

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_KEY).build()

    # voice input
    application.add_handler(
        MessageHandler(filters.VOICE & ~filters.COMMAND, voice, block=True)
    )

    # text input
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_input, block=True))
    application.add_handler(
        MessageHandler(filters.PHOTO & ~filters.COMMAND, photo, block=True)
    )

    # commands
    application.add_handler(CommandHandler("n_words", n_words))
    application.add_handler(CommandHandler("clear", clear, block=False))
    application.add_handler(CommandHandler("listen", listen, block=False))
    application.add_handler(CallbackQueryHandler(button))


    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
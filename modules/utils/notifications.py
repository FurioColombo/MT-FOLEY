import requests
from config.config import Config

def notify_telegram(message, verbose=False):
    config = Config.get_telegram_config()
    apiToken = config.apiToken
    chatID = config.chatID
    if apiToken is str and chatID is str:
        apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

        try:
            response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
            if verbose:
                print(f'Telegram message response: {response}.')
            if response.ok is not True:
                print(f'TELEGRAM NOTIFICATION ERROR: response is not OK: {str(response)}.')
        except Exception as e:
            print(e)

from twilio.rest import Client
from secrets import *
def sendMessage(message):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message = client.messages \
        .create(
            body=message,
            from_= SENDER,
            to= RECEIVER
        )
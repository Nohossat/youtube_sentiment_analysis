# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from nohossat_cas_pratique.user_creation import save_database, create_random_pwd


def sendgrid_email(message):
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
        print("Email sent")
        return 'Email sent'
    except Exception as e:
        print(e)
        return f'Cannot send the email - error : {e}'


def send_email_user_creation(username, recipient):
    password = create_random_pwd()

    save_database(username, recipient, password)

    msg = f"Here is the password to use the Sentiment analysis tool. Password : {password}. \n Please use the user you passed while creating your account to access the special features of the API."

    message = Mail(
        from_email=os.getenv('SENDGRID_SENDER'),
        to_emails=recipient,
        subject='Sentiment Analysis API - User creation',
        html_content=msg)

    return sendgrid_email(message)


def send_email(url, recipient):
    msg = f"The experiment is done, you can see the results at this url : <a href='{url}' target='_blank'>here</a>"

    message = Mail(
        from_email=os.getenv('SENDGRID_SENDER'),
        to_emails=recipient,
        subject='Sentiment Analysis API - Results',
        html_content=msg)

    return sendgrid_email(message)

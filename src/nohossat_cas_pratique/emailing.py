# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


def send_email(url, recipient):
    msg = f"The experiment is done, you can see the results at this url : <a href='{url}' target='_blank'>here</a>"

    message = Mail(
        from_email=os.getenv('SENDGRID_SENDER'),
        to_emails=recipient,
        subject='Sentiment Analysis API - Results',
        html_content=msg)

    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
        print("Email sent")
    except Exception as e:
        print(e)

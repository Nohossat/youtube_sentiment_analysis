import os

from nohossat_cas_pratique.emailing import send_email, send_email_user_creation, sendgrid_email
from nohossat_cas_pratique.user_creation import delete_user

from sendgrid.helpers.mail import Mail


def test_sendgrid_email():
    msg = "hola"
    message = Mail(
        from_email=os.getenv('SENDGRID_SENDER'),
        to_emails=os.getenv('SENDGRID_SENDER'),
        subject='Sentiment Analysis API - Test',
        html_content=msg)

    response = sendgrid_email(message)
    assert response == "Email sent"


def test_send_email():
    print('hello')
    response = send_email("https://neptune.ai", os.getenv('SENDGRID_SENDER'))
    assert response == "Email sent"


def test_send_email_user_creation():
    response = send_email_user_creation("test_user", os.getenv('SENDGRID_SENDER'))
    delete_user("test_user")
    assert response == "Email sent"

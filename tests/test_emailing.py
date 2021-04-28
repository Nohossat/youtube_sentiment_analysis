import os

from nohossat_cas_pratique.emailing import send_email


def test_send_email():
    print('hello')
    response = send_email("https://neptune.ai", os.getenv('SENDGRID_SENDER'))
    assert response == 202

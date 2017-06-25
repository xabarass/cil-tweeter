import requests

DOMAIN_NAME='serber.club'
API_KEY="key-1d0fd7a2a537e88720365261c20dfe78"

class Emailer:
    def __init__(self, subject, destination_email=None):
        self.destination=destination_email
        self.subject=subject
        self.url = 'https://api.mailgun.net/v3/{}/messages'.format(DOMAIN_NAME)

    def send_report(self, message):
        if self.destination is None:
            return

        auth = ('api', API_KEY)
        data = {
            'from': 'CIL training results <emailer@{}>'.format(DOMAIN_NAME),
            'to': self.destination,
            'subject': self.subject,
            'text': message,
        }

        response = requests.post(self.url, auth=auth, data=data)
        response.raise_for_status()
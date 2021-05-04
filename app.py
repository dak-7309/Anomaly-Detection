from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server
from classifier import main
app = Flask(__name__)


def predict():
    file =  file_upload("Select a video:", accept="video/*")
	f = open('temp.mp4','wb')
	f.write(file['content'])
	result = main('temp.mp4')
	put_text(result)

app.add_url_rule('/tool', 'webio_view', webio_view(predict),methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    start_server(predict, port=args.port)
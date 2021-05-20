from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server
from classifier import main
import time
import plotly.graph_objects as go
app = Flask(__name__)


def predict():
	animals=['Assult','Normal','Arrest','Explosion','Vandalism']
	file = file_upload("Select a video:", accept="video/*")
	start = time. time()
	with put_loading():
		f = open('temp.mp4', 'wb')
		f.write(file['content'])
		result,list_final = main('temp.mp4')
		# tx = "Prediction: "+str(result)
    #Prediction: "+str(result)
		img = open('output_img.jpg', 'rb').read() 
		style(put_text('Middle frame from Video'), 'text-align: center')
		style(put_image(img, width='500px'), 'display: block; margin-left: auto; margin-right: auto')
		fig = go.Figure([go.Bar(x=animals, y=list_final)])
		html = fig.to_html(include_plotlyjs="require", full_html=False)
		put_text('\n')
		style(put_text('Prediction Graph'), 'text-align: center')
		style(put_html(html), 'margin: auto')

		end = time. time()	
		print ("Time elapsed:", end - start)
		print(list_final)
		# put_text(tesxt)
		
		style(put_text('Prediction:'), 'text-align: center')
		style(put_text(str(result)), 'font-size: 200%;text-align: center')


app.add_url_rule('/tool', 'webio_view', webio_view(predict),methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--port", type=int, default=8080)
	args = parser.parse_args()
	start_server(predict, port=args.port)

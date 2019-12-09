from flask import Flask, jsonify, request, Response
import sys
import os
sys.path.insert(1, '..')
from text_class_learner import MultiLabelTextClassifier
from data_utils.json_loader import JSONLoader

# Wizaron/pytorch-cpp-inference --> C++ example dockerfile server

app = Flask(__name__)
# init model
TextClassifier = None #MultiLabelTextClassifier.load(os.path.join(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME))
DocLoader = JSONLoader()

@app.route('/')
def hello_world():

	return jsonify({'message':'App started'})

@app.route('/predict', methods=['POST'])
def predict():
	global TextClassifier
	if TextClassifier is None:
		return jsonify({'error':'Model not yet initialized, please call load first.'})
	if request.method == 'POST':
		text = request.args.get('text', '')
		if len(text.strip()) == 0:
			return Response(status=400)

		print(text)
		# get preds
		preds, word_attention_scores, sent_attenion_scores, loss = TextClassifier.predict_text(text, return_doc=False)

		# convert preds to label names
		pred_labels = TextClassifier.pred_to_labels(preds)

		return jsonify({'pred':pred_labels})


@app.route('/load', methods=['POST'])
def load_model():
	os.chdir('..')
	global TextClassifier
	# Load a locally saved model
	if request.method == 'POST':
		try:
			print(request)
			p = request['path']
			x = 1
		except KeyError:
			return jsonify({'error':'A local path must be given to load a new model'})

		# try:
		TextClassifier = MultiLabelTextClassifier.load(p)
		# except Exception as e:
		# 	return jsonify({'error':'Loading giving path failed.', 'message':str(e)})

		return jsonify({'status':'OK'})
	else:
		return Response(status=501)


@app.route('/download_model', methods=['POST'])
def download_model():
	#TODO: agree on how/where to download models from
	None


if __name__ == '__main__':
    app.run(debug=True)
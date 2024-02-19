import flask
from flask import request

from Ecg import ECG

ecg = ECG()
app = flask.Flask(__name__)


@app.route('/')
def home():
    return "Ecg Insights"


@app.route('/predict', methods=['POST'])
def handle_request():
    if 'image' not in request.files:
        return "no image is provided"

    image = request.files['image']

    """#### **DIVIDING LEADS**"""
    # call the Divide leads method
    dividing_leads = ecg.DividingLeads(image)

    """#### **EXTRACTING SIGNALS(1-12)**"""
    # call the signal extraction method
    ecg.SignalExtraction_Scaling(dividing_leads)

    """#### **CONVERTING TO 1D SIGNAL**"""
    # call the combine and conver to 1D signal method
    ecg_1dsignal = ecg.CombineConvert1Dsignal()

    """#### **PERFORM DIMENSIONALITY REDUCTION**"""
    # call the dimensionality reduction function
    ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)

    """#### **PASS TO PRETRAINED ML MODEL FOR PREDICTION**"""
    # call the Pretrained ML model for prediction
    # result = {'prediction': ecg.ModelLoad_predict(ecg_final)}

    return ecg.ModelLoad_predict(ecg_final)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
#if __name__ == '__main__':
#    from gunicorn import config
#    from gunicorn.workers.sync import SyncWorker

#    class CustomWorker(SyncWorker):
#        def run(self):
#            config._use_forwarded_for = True  # Enable if running behind a proxy
#            super().run()

#    config.Worker = "app:CustomWorker"

#    options = {
#        'bind': '0.0.0.0:5000',
#        'workers': 4,  # You can adjust the number of workers
#    }

#    from gunicorn.app.wsgiapp import WSGIApplication
#    WSGIApplication("%(prog)s [OPTIONS]").run()

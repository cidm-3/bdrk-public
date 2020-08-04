"""
Script for serving.
"""
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import models 
import pandas as pd
import numpy as np
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
from flask import Flask, Response, current_app, request

OUTPUT_MODEL_NAME = "/artefact/model_COVID_bdrk.h5"


# def predict_prob(subscriber_features,
#                  model=pickle.load(open(OUTPUT_MODEL_NAME, "rb"))):
#     """Predict churn probability given subscriber_features.

#     Args:
#         subscriber_features (dict)
#         model

#     Returns:
#         churn_prob (float): churn probability
#     """
#     row_feats = list()
#     for col in SUBSCRIBER_FEATURES:
#         row_feats.append(subscriber_features[col])

#     for area_code in AREA_CODES:
#         if subscriber_features["Area_Code"] == area_code:
#             row_feats.append(1)
#         else:
#             row_feats.append(0)

#     for state in STATES:
#         if subscriber_features["State"] == state:
#             row_feats.append(1)
#         else:
#             row_feats.append(0)

#     # Score
#     churn_prob = (
#         model
#         .predict_proba(np.array(row_feats).reshape(1, -1))[:, 1]
#         .item()
#     )

#     # Log the prediction
#     current_app.monitor.log_prediction(
#         request_body=json.dumps(subscriber_features),
#         features=row_feats,
#         output=churn_prob
#     )

#     return churn_prob


# pylint: disable=invalid-name
app = Flask(__name__)

def load_nn_model(self): 
        #loaded_model = tf.keras.models.load_model('model')
        # load json and create model
        loaded_model = models.load_model(OUTPUT_MODEL_NAME)
        # Pre-Covid:  models.load_model('model/model.h5', custom_objects={"weighted_mse": self.weighted_mse})
        return loaded_model
        
def model_predict(self, loaded_model, X_new):
        Y_pred = pd.DataFrame(loaded_model.predict(X_new))
        Y_pred[Y_pred<0] = 0
        Y_pred.columns = ['beyond_minus6.0H', 'minus6.0_to_5.75H', 'minus5.75_to_5.5H',
                           'minus5.5_to_5.25H', 'minus5.25_to_5.0H', 'minus5.0_to_4.75H',
                           'minus4.75_to_4.5H', 'minus4.5_to_4.25H', 'minus4.25_to_4.0H',
                           'minus4.0_to_3.75H', 'minus3.75_to_3.5H', 'minus3.5_to_3.25H',
                           'minus3.25_to_3.0H', 'minus3.0_to_2.75H', 'minus2.75_to_2.5H',
                           'minus2.5_to_2.25H', 'minus2.25_to_2.0H', 'minus2.0_to_1.75H',
                           'minus1.75_to_1.5H', 'minus1.5_to_1.25H', 'minus1.25_to_1.0H',
                           'minus1.0_to_0.75H', 'minus0.75_to_0.5H', 'minus0.5_to_0.25H',
                           'minus0.25_to_0H']
        return Y_pred

@app.route("/", methods=["POST"])
def predict(): 
    df = pd.read_csv(request.files.get('file'))
    loaded_model = load_nn_model() 
    Y_pred = model_predict(loaded_model, df)
    return Y_pred.to_json()



# def get_churn():
#     """Returns the `churn_prob` given the subscriber features"""

#     subscriber_features = request.json
#     result = {
#         "churn_prob": predict_prob(subscriber_features)
#     }
#     return result


# @app.before_first_request
# def init_background_threads():
#     """Global objects with daemon threads will be stopped by gunicorn --preload flag.
#     So instantiate them here instead.
#     """
#     current_app.monitor = ModelMonitoringService()


# @app.route("/metrics", methods=["GET"])
# def get_metrics():
#     """Returns real time feature values recorded by prometheus
#     """
#     body, content_type = current_app.monitor.export_http(
#         params=request.args.to_dict(flat=False),
#         headers=request.headers,
#     )
#     return Response(body, content_type=content_type)


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()

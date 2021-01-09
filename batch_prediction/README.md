# Batch prediction examples

## Introdcution

[Cloud AI Platform](https://cloud.google.com/ai-platform) (CAIP) provides a serverless platform for training and serving machine learning (ML) models. You can use [the batch prediction](https://cloud.google.com/ai-platform/prediction/docs/batch-predict) when you have a large number of instances to get predictions. You store prediction input files in the storage bucket and submit a batch prediction job. The prediction results are recorded in text files and stored in the storage bucket.

When using the batch prediction, you need to consider the fact that the order of prediction results in the output files can be different from the instances in the prediction input files. It is because the batch prediction is conducted with multiple workers in a distributed manner. Hence you need to modify your ML model so that it accepts a unique identifier as a part of the input features, and outputs the same identifier as a part of the prediction result. Conceptually, this can be illustrated as in the following diagram:

```
                ML Model
              ------------
             |            |
     key ----|------------|---- key
             |            |
         ----|            |
features ----|            |---- prediction
         ----|            |
             |            |
              ------------

```

The followings are examples of an input and the corresponding output.

Input file:
```
{"features": [47.0, 14.5, 1.0, 0.0, 0.0, 0.0, 1.0], "key": 0}
{"features": [38.0, 227.525, 1.0, 0.0, 1.0, 0.0, 0.0], "key": 1}
{"features": [24.0, 83.1583, 1.0, 0.0, 1.0, 0.0, 0.0], "key": 2}
{"features": [19.0, 8.05, 0.0, 1.0, 0.0, 0.0, 1.0], "key": 3}
{"features": [34.0, 23.0, 1.0, 0.0, 0.0, 1.0, 0.0], "key": 4}
{"features": [23.0, 11.5, 0.0, 1.0, 0.0, 1.0, 0.0], "key": 5}
{"features": [24.0, 263.0, 1.0, 0.0, 1.0, 0.0, 0.0], "key": 6}
{"features": [17.0, 7.2292, 0.0, 1.0, 0.0, 0.0, 1.0], "key": 7}
{"features": [41.0, 39.6875, 1.0, 0.0, 0.0, 0.0, 1.0], "key": 8}
{"features": [17.0, 16.1, 1.0, 0.0, 0.0, 0.0, 1.0], "key": 9}
```

Output file:
```
{"key": 0, "prediction": [0.30424559116363525]}
{"key": 1, "prediction": [0.9568923115730286]}
{"key": 2, "prediction": [0.7721445560455322]}
{"key": 3, "prediction": [0.17484110593795776]}
{"key": 4, "prediction": [0.4179154634475708]}
{"key": 5, "prediction": [0.18336786329746246]}
{"key": 6, "prediction": [0.9531108736991882]}
{"key": 7, "prediction": [0.18050001561641693]}
{"key": 8, "prediction": [0.4109613597393036]}
{"key": 9, "prediction": [0.4729471206665039]}
```

In this example, integer keys are used as a unique identifier and both input and output are sorted by the key. You can match the input and output using the key value even if the output is randomly ordered.

## Modify Keras models

There are multiple ways to add a unique key to the model. If you use Keras to define a model, you can use one of the following methods.

1. Use the functional API to create a wrapper model that adds a key field to an existing model.
2. Use `@tf.function` decorator to define a wrapper function to make predictions with keys.

### Using the functional API

Suppose that you defined and trained a Keras model. The model object is stored in the variable `model`. You can define and compile a wrapper model `wrapper_model` as below:

```
key = layers.Input(shape=(), name='key', dtype='int32')
pred = layers.Concatenate(name='prediction_with_key')(
    [model.output, tf.cast(layers.Reshape((1,))(key), tf.float32)])
wrapper_model = models.Model(inputs=[model.input, key], outputs=pred)
wrapper_model.compile()
```

You export `wrapper_model` in the saved_model format and deploy it to the AI Platform. The following code snippet shows how you use the deployed model to make an online prediction. The model accepts the `key` field in addition to `features`.

```
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json

credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials, cache_discovery=False)

request_data =  {'instances':
  [
    {"features": [47.0, 14.5, 1.0, 0.0, 0.0, 0.0, 1.0], "key": 0},
    {"features": [38.0, 227.525, 1.0, 0.0, 1.0, 0.0, 0.0], "key": 1},
  ]
}

parent = 'projects/%s/models/%s/versions/%s' % (PROJECT, 'yourmodel', 'v1')
response = api.projects().predict(body=request_data, name=parent).execute()
print(json.dumps(response, sort_keys = True, indent = 4))
```

The output contains the predicion and key values as below:

```
{
    "predictions": [
        {
            "prediction_with_key": [
                0.10660852491855621,
                0.0
            ]
        },
        {
            "prediction_with_key": [
                0.7562407851219177,
                1.0
            ]
        }
    ]
}
```

The [Notebook](batch_prediction_with_Keras.ipynb) explains the whole procedure to use this method for the batch prediction. Follow [Use JupyterLab Notebooks](README.md#use-jupyterlab-notebooks) to run the notebook.


## Use JupyterLab Notebooks

1. Create a new GCP project and enable the following API. 
- AI Platform Training & Prediction API
- Notebooks API
- Dataflow API

2. Launch a new notebook instance from the "AI Platform -> Notebooks" menu. Choose "TensorFlow Enterprise 2.3 without GPUs" for the instance type.

3. Open JupyterLab and execute the following commond from the JupyterLab terminal.

```
git clone https://github.com/enakai00/CAIP-examples
```

4. Open the following notebooks and follow the instruction.

- `CAIP-examples/batch_prediction/batch_prediction_with_Keras.ipynb`: Using the functional API.
- `CAIP-examples/batch_prediction/batch_prediction_with_Keras2.ipynb`: Using the `@tf.function` decorator.

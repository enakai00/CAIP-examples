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



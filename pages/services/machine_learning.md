---
layout: frontpage
title: Machine Learning
---

<div class="navbar">
  <div class="navbar-inner">
      <ul class="nav">
          <li><a href="python_development.html">python development</a></li>
          <li><a href="backend_development.html">backend development</a></li>
      </ul>
  </div>
</div>

## Machine Learning

#### Generate value from your data!
<img src="../../assets/services/machine_learning_1.png" alt="machine learning" title="machine learning"/>
<br/>
<br/>

***
Modern systems and devices generate more data than never before and the tendency is to keep increasing. In previous years it was not feasible to extract value out of that data, but recent storage and processing price decreases are enabling new applications and business opportunities.

There are a wide range of applications:

- Machinery cost reduction. Replace expensive sensors with models relying on available data
- Anomaly detection. Detect in real time malfunctioning of devices based on historical data. The same principle can be applied to analyze operations, such as identifying fraud and misuse of products or services
- Segmentation and Classification. Classify a population based on data they generate
- Image Recognition. Customized image recognition for complex tasks


***

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from models.model import Model


class NeuralNetworkModel(Model):
    """
    A class to build a neural network and measure its execution time.
    """

    def build_model(self, layers=(20,20,0,0)):
        """
        Build and train a model with the provided characteristics.
            :param self: 
            :param layers=(20,20,0,0): number of neurons
        """

        # build the pipeline
        self.pipeline = \
            Pipeline([('scl', StandardScaler()),
                      ('poly_features', PolynomialFeatures(degree=self.degree)),
                      ('estimator', KerasRegressor(
                        build_fn=self._base_model(layers=layers),
                        epochs=1, batch_size=50, verbose=0))])
        # train the model
        self._train_model()
```
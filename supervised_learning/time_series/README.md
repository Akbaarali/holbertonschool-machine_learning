# Forecasting Bitcoin Price with Time Series and RNNs

![BTC Forecasting Graph / Model Performance Image Here]

## Introduction

For this project, I worked on a time series forecasting problem using Bitcoin price data. The main goal was to use the past 24 hours of BTC data to predict the Bitcoin closing price for the next hour.

Bitcoin prices change over time, so this is not a normal classification problem. It is a time series problem because the order of the data matters. The price at one moment is connected to what happened before it.

To solve this problem, I used a neural network model with an RNN-based architecture. RNNs are useful for this type of problem because they can learn from sequential data.

The final files for this project were:

* `preprocess_data.py`
* `forecast_btc.py`

---

## What is Time Series Forecasting?

Time series forecasting is a method used to predict future values based on past values.

A time series is data that is ordered by time. Some examples are:

* Stock prices
* Weather temperature
* Bitcoin price
* Website traffic
* Electricity usage

In this project, each row of the dataset represents a 60-second time window for Bitcoin trading. Because the rows follow time order, the model has to learn patterns from previous time steps.

The task was to use the previous 24 hours of BTC data to predict the BTC closing price for the next hour.

Since each row represents one minute, 24 hours equals:

```text
24 hours × 60 minutes = 1440 time steps
```

So, each input sample contains 1440 minutes of previous BTC data.

---

## Dataset

The project used the **Coinbase** and **Bitstamp** Bitcoin datasets.

Each row represents a 60-second time window and contains information about BTC trading during that minute.

The main columns in the dataset are:

| Column          | Meaning                                       |
| --------------- | --------------------------------------------- |
| Timestamp       | Start time of the time window in Unix time    |
| Open            | BTC price at the beginning of the time window |
| High            | Highest BTC price during the time window      |
| Low             | Lowest BTC price during the time window       |
| Close           | BTC price at the end of the time window       |
| Volume BTC      | Amount of BTC traded                          |
| Volume Currency | Amount of USD traded                          |
| Weighted Price  | Volume-weighted average price                 |

The target value I wanted to predict was the future BTC **Close** price.

---

## Preprocessing Method

Before training the model, I had to preprocess the raw data.

The dataset was raw, so I could not use it directly. First, I loaded the data using Pandas. Then I cleaned it and prepared it for a time series model.

The preprocessing steps I used were:

1. Load the Coinbase and Bitstamp datasets.
2. Remove missing or invalid values.
3. Sort the data by timestamp.
4. Select useful features.
5. Scale the data.
6. Create input sequences using the past 24 hours.
7. Create target values using the next hour close price.
8. Save the processed data for the model.

I chose this preprocessing method because the model needs clean and ordered data. If the time order is wrong, the model will learn incorrect patterns.

I also scaled the data because Bitcoin price values and volume values can be very different in size. Scaling helps the neural network train more smoothly and prevents large values from dominating smaller values.

---

## Features Used

For the model input, I used the useful numerical features from the dataset.

The features included:

* Open price
* High price
* Low price
* Close price
* BTC volume
* Currency volume
* Weighted price

I did not use the timestamp directly as a normal feature because the timestamp itself is just a large number. Instead, it was mainly used to keep the data in the correct order.

The model input shape was based on:

```text
1440 previous time steps × number of selected features
```

This means the model looks at the last 24 hours of BTC data before making a prediction.

---

## Using tf.data.Dataset

For feeding the data into the model, I used `tf.data.Dataset`.

This is useful because it helps TensorFlow handle the data more efficiently during training.

The general process was:

```python
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

I used `tf.data.Dataset` because it makes the training pipeline cleaner and faster. It also helps with batching and prefetching data.

Batching means the model does not train on the whole dataset at once. Instead, it trains using smaller groups of samples.

Prefetching helps prepare the next batch while the model is training on the current batch.

---

## Model Architecture

For the model, I used an RNN-based neural network.

The main reason I chose an RNN architecture is because Bitcoin price data is sequential. The previous prices can affect future prices, so the model needs to understand patterns over time.

A simple version of the model architecture is:

```python
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
```

The model contains:

* LSTM layer
* Dropout layer
* Another LSTM layer
* Another Dropout layer
* Dense output layer

I used **LSTM** because it is a type of RNN that is better at learning longer-term dependencies. Since the model uses the past 24 hours of data, LSTM is a good choice.

I used **Dropout** to help prevent overfitting. This makes the model more general and less likely to memorize the training data.

The final Dense layer outputs one value, which is the predicted BTC closing price.

---

## Loss Function and Optimizer

The project required the model to use **Mean Squared Error** as the cost function.

I used MSE because this is a regression problem. The model is not predicting a class. It is predicting a continuous value, which is the BTC price.

The model was compiled like this:

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

I used the Adam optimizer because it is a strong general-purpose optimizer and usually works well for neural networks.

I also used MAE as an additional metric because it gives a more understandable error value.

---

## Training the Model

After preprocessing the data and creating the datasets, I trained the model on the training set and validated it on the validation set.

During training, the model tried to reduce the MSE loss. This means it tried to make the predicted BTC price closer to the real BTC closing price.

I also used validation data to check whether the model was learning general patterns or just memorizing the training data.

The training process helped me compare the training loss and validation loss.

If the training loss decreases but the validation loss becomes worse, that can be a sign of overfitting.

---

## Results

After training, I checked the model performance using the validation results.

The main results were:

| Metric          | Value                              |
| --------------- | ---------------------------------- |
| Training Loss   | [Insert your training loss here]   |
| Validation Loss | [Insert your validation loss here] |
| Training MAE    | [Insert your training MAE here]    |
| Validation MAE  | [Insert your validation MAE here]  |

I also created graphs to visualize the model performance.

The first graph showed the training loss and validation loss over time. This helped me see whether the model was improving during training.

The second graph compared the predicted BTC prices with the real BTC prices. This helped me understand how close the model predictions were to the actual values.

Example graph labels:

```text
Training Loss vs Validation Loss
Actual BTC Price vs Predicted BTC Price
```

From the results, I observed that the model was able to learn some general price movement patterns. However, Bitcoin price forecasting is still very difficult because BTC is highly volatile and can change suddenly due to market news, investor behavior, and global events.

---

## Challenges

One challenge in this project was preparing the data correctly.

For time series forecasting, data order is very important. If the rows are shuffled before creating sequences, the model will not learn correctly.

Another challenge was the size of the input data. Since each input uses 1440 time steps, training can become slower than a normal model.

Also, Bitcoin is very hard to forecast perfectly. Even if the model learns from past data, future prices can still be affected by events that are not included in the dataset.

---

## What I Learned

From this project, I learned that time series forecasting is different from normal supervised learning.

In normal supervised learning, each row can often be treated independently. But in time series forecasting, the order of the rows matters a lot.

I also learned how to prepare sequential data for an RNN model. The model needs input windows from the past and target values from the future.

Another important thing I learned was how useful `tf.data.Dataset` can be for training TensorFlow models. It makes the data pipeline more organized and efficient.

---

## Conclusion

In this project, I used Bitcoin trading data to create a time series forecasting model.

The model used the previous 24 hours of BTC data to predict the closing price for the next hour.

The main steps were:

* Load the raw BTC datasets
* Clean and preprocess the data
* Scale the features
* Create 24-hour time windows
* Use `tf.data.Dataset` to feed data to the model
* Build and train an RNN/LSTM model
* Evaluate the model using MSE
* Plot the model performance

Overall, this project helped me understand how RNNs can be used for time series forecasting.

---

## Final Thoughts

Forecasting Bitcoin is not easy because the price is very unstable. A model can learn patterns from the past, but it cannot know unexpected future events.

Still, this project was useful because it showed me how machine learning can be applied to financial time series data.

For future improvements, I could try:

* More advanced LSTM or GRU models
* More features
* More training data
* Better normalization methods
* Different window sizes
* Comparing the model with a simple baseline

The GitHub repository for this project can be found here:

[Insert your GitHub link here]

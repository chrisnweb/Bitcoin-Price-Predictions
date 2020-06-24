# Bitcoin-Price-Predictions
## **Build Bitcoin Sentiment and Price Predictions Using Machine Learning/Deep Learning**

***Disclaimer:*** *All the information in this repository, README including the algorithm and Neural Network were provided and published for educational purpose only, not for investment nor investment advice.*

### **Introduction/Description of project**

Bitcoin (BTC) is the first decentralized digital currency and a payment system that was invented by an unknown person whose name is Satoshi Nakamoto from Japan, invented in 2008 and released it as open source software in 2009.

### **Goal**
*	First, to collect the bitcoin raw tweets from Twitter for price prediction using sentiment analysis. Sentiment analysis is one of the hottest and trending topics in machine learning.
* Second, to collect the historical price data from Yahoo Finance and use a simple Neural Network to predict future prices of bitcoin for a short period of time.

### **The Neural Network methods used**
* ***Convolutional Neural Network (CNN)*** – It is a class of deep neural networks which is commonly applied visual imagery.
* ***Recurrent Neural Network (RNN)*** – It is a class of artificial neural networks. RNN is used for analysis of sequential data, i.e. time series data prediction. It has proved to be one of the most powerful models for processing sequential data. 
* ***Long Short-Term memory (LSTM) network*** – It is a type of deep learning used for analysis of sequential data, i.e. time series data prediction. It is also a unit and a special kind of RNN. LSTM is explicitly designed to avoid the long-term dependency problem.

Thus, we can say that RNN is more helping us in data processing predicting our next step whereas CNN helps us in visual analyzing.
I have decided to use CNN + LSTM on the BTC tweets and RNN + LSTM on BTC historical price data.

### **Load Packages**
Load all required python packages, i.e. TextBlob, WordCloud, stopwords, etc.

### Collect BTC tweets
##### (Refer to Jupyter Notebook file “bitcoin_tweets_download_05312020.ipynb”)

* To collect the fresh raw tweets, I followed the example Tweet API from the previous class MSDS 600. 
* To restrict the topic on bitcoin, I filtered the word “bitcoin” only. I have collected 7895 raw tweets in total.
* Saved the raw tweets to csv file.

### **Sentiment analysis on the raw tweets**
##### (Refer to Jupyter Notebook file “rawtweets_sentiment_05312020.ipynb”)
Generate the TextBlob package to obtain the polarity, sensitivity and sentiment. 

### Pre-Processing – Exploratory Data Analysis (EDA)
##### (Refer to Jupyter Notebook file “btctweets_pre-processing-05312020.ipynb”)

1.	Created and loaded the new csv file named “btctweets_05312020.csv” – Randomly used only 2020 (out of 7895) tweets with 6 columns (date_time, name, text, polarity, sensitivity and sentiment)

2.	Cleaned the tweets (text) with Stopwords and WordPunctTokenizer

3.	Visualization
      * Word Cloud – Generated a word cloud use WordCloud python package.
     As expected, the “bitcoin” is one of the most frequent word shown in the tweets.

      * Created a senti ent mood count with Countplot in Seaborn.

      * Out of the 2020 tweets, 964 tweets are classified as neutral, 788 tweets are classified as positive, and 268 tweets are classified as negative.

4.	Export the clean tweet data 
     * Saved the clean tweets (text) as “cleaned_tweet_data.csv”
          * Saved a final clean tweets data: “cleaned_tweet_data_05312020.csv”


## Sentiment Analysis Prediction – CNN-LSTM

### 1.	Data split – training and testing

#### •	80% training data and 20 test data

### 3.	Model

#### •	Build our neural network model with 1D-Convolution kernel and LSTM in Keras + add arguments embedding size and input length

##### -	We use 1D-Conv kernel to extract the information
##### -	Filters are 32, kernel_size is 3, activation is ‘ReLU’ and set padding equal to ‘same”
##### -	1D-MaxPooling is used after 1D-Conv
##### -	96-unit LSTM is used for signal classification.
##### -	Here, we also create additional input layer with num_classes split into 4 and 2 nodes (for better chance of success. We use ReLU activation and Dropout of 20% (this to prevent model from overfitting)	 
##### -	Lastly, we add an output layer with num_classes and use softmax activation.

#### •	Compile the model
##### -	The loss parameter is specified as type ‘categorical_crossentropy’
##### -	The metrics parameter is set to ‘accuracy’
##### -	Lastly, we use the ‘adam’ optimizer for training the network.

#### •	Training the model
##### The epoch is set to 25. Since the dataset has 2020 samples (small size dataset), to be safe we are using a batch size of 64. 
##### (Note: I even have used batch size of 32.  It won’t make a huge difference for our problem)
##### The model training is done in one single method ‘fit’.
 
#### •	Examining the performance 
##### -	The output test accuracy of 80%, is acceptable to us. What it means to us is that 20% of cases would not be classified correctly.

##### -	Plotting loss metrics
##### As you can see in the diagram, the loss on the training set decreases rapidly in the first 8 epochs. The loss on the validation decrease slightly in the first 6 epochs and increase rapidly on the next 6 epochs. 
 	 
##### -	Plotting accuracy metrics
##### As you can see in the diagram, the accuracy increases rapidly in the first 8 epochs, indicating that the network is learning fast. The remaining curve are flattens indicating that not many epochs are required to train the model further.

### 4.	Prediction/Evaluation
#### Confusion matrix is used to evaluate the quality of the output of classifier on the test set.

### 5.	Results
#### Training model shows 80% and confusion matrix on the test set is also shows 80% accuracy. It means the confusion matrix is showing good performance. The prediction result is good.

## Conclusion
#### Out of the 2020 tweets, 964 tweets are classified as neutral, 788 tweets are classified as positive, and 268 tweets are classified as negative. 13% are negative while 39% are positive. There are more positive than negative. Based on this sentiment results on the tweets, we can predict that the BTC price will go up in the near future.


## Reference Sources

#### Irene Lui. 2018. Twitter-US-Airline-Sentiment_J2D_Project_Python. https://github.com/ireneliu521/Twitter-US-Airline-Sentiment_J2D_Project_Python/blob/master/Twitter%20US%20Airline%20Sentiment.ipynb. (2020).

#### Omer Berat Sezer. 2018. LSTM_RNN_Tutorials_with_Demo. https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo. (2020)

#### stackoverflow.com (2017). How to plot confusion matrix correctly. https://stackoverflow.com/questions/44189119/how-to-plot-confusion-matrix-correctly. Last accessed: 2020-06-19

```


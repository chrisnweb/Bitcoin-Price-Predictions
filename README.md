# Bitcoin-Price-Predictions
## **Build Bitcoin Sentiment and Price Predictions Using Machine Learning/Deep Learning**

***Disclaimer:*** *All the information in this repository, README including the algorithm and Neural Network were provided and published for educational purpose only, not for investment nor investment advice.*

### **Introduction/Description of project**

Bitcoin (BTC) is the first decentralized digital currency and a payment system that was invented by an unknown person whose name is Satoshi Nakamoto from Japan, invented in 2008 and released it as open source software in 2009.

#### **Goal**
*	First, to collect the bitcoin raw tweets from Twitter for price prediction using sentiment analysis. Sentiment analysis is one of the hottest and trending topics in machine learning.
* Second, to collect the historical price data from Yahoo Finance and use a simple Neural Network to predict future prices of bitcoin for a short period of time.

#### **The Neural Network methods used**
* ***Convolutional Neural Network (CNN)*** – It is a class of deep neural networks which is commonly applied visual imagery.
* ***Recurrent Neural Network (RNN)*** – It is a class of artificial neural networks. RNN is used for analysis of sequential data, i.e. time series data prediction. It has proved to be one of the most powerful models for processing sequential data. 
* ***Long Short-Term memory (LSTM) network*** – It is a type of deep learning used for analysis of sequential data, i.e. time series data prediction. It is also a unit and a special kind of RNN. LSTM is explicitly designed to avoid the long-term dependency problem.

Thus, we can say that RNN is more helping us in data processing predicting our next step whereas CNN helps us in visual analyzing.
I have decided to use CNN + LSTM on the BTC tweets and RNN + LSTM on BTC historical price data.

### **Load Packages**
Load all required python libraries/packages, i.e. TextBlob, WordCloud, NLTK stopwords, Keras, TensorFlow, etc.

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



## Sentiment Analysis Evaluation/Prediction – CNN-LSTM

### 1.	Dataset
   * Loading the ‘cleaned tweet data” and converting it to a pandas dataframe.

### 2.	EDA
   * Plot/visualize the mood count
   * Text length
   * Visualization with WordClould: neutral, positive and negative sentiment.

### 3.    Data split – training and testing ###
   * 80% training data and 20 test data

### 4.	Tokenize 
   * The tokenize can help us build a NN model based on words.
   * Tokenize the text with ‘text_to_sequences’.

### 5.    Model
  * **Build our neural network model with 1D-Convolution kernel and LSTM in Keras + add arguments embedding size and input length.**       
     * We use 1D-Conv kernel to extract the information.         
     * Filters are 32, kernel_size is 3, activation is ‘ReLU’ and set padding equal to ‘same'       
     * 1D-MaxPooling is used after 1D-Conv        
     * 96-unit LSTM is used for signal classification.      
     * Here, we also create additional input layer with num_classes split into 4 and 2 nodes (for better chance of success. We use ReLU activation and Dropout of 20% (this to prevent model from overfitting)              
     * Lastly, we add an output layer with num_classes and use softmax activation.        

* **Compile the model**
   - The loss parameter is specified as type ‘categorical_crossentropy’        
   - The metrics parameter is set to ‘accuracy’       
   - Lastly, we use the ‘adam’ optimizer for training the network.        

* **Fitting/Training the CNN model**                           
  * The epoch is set to 25. Since the dataset has 2020 samples (small size dataset), to be safe we are using a batch size of 64. 
     (Note: I even have used batch size of 32.  It won’t make a huge difference for our problem)         
  * The model training is done in one single method ‘fit’.
 
* **Examining the performance**          
   - The output test accuracy of 80%, is acceptable to us. What it means to us is that 20% of cases would not be classified correctly.
   - **Plotting loss metrics**
   ![](https://github.com/chrisnweb/Bitcoin-Price-Predictions/blob/master/Images/lossmetrics.png)
               As you can see in the diagram, the loss on the training set decreases rapidly in the first 8 epochs. The loss on the        validation decrease slightly in the first 6 epochs and increase rapidly on the next 6 epochs.  
               
   - **Plotting accuracy metrics**
   ![](https://github.com/chrisnweb/Bitcoin-Price-Predictions/blob/master/Images/accmetrics.png)
               As you can see in the diagram, the accuracy increases rapidly in the first 8 epochs, indicating that the network is learning fast. The remaining curve are flattens indicating that not many epochs are required to train the model further.

### 6.	Prediction/Evaluation
   * Confusion matrix is used to evaluate the quality of the output of model classifier.
   * Run a predict class with “model.predict_classes” on the test set.
   * Calculate the confusion matrix, precision, and f1-score


### 7.	Results
Training model shows 80% and confusion matrix on the test set is also shows 80% accuracy. It means the confusion matrix is showing good performance. The prediction/evaluation result is good.          
Out of the 2020 tweets, 964 tweets are classified as neutral, 788 tweets are classified as positive, and 268 tweets are classified as negative. 13% are negative while 39% are positive. There are more positive than negative. Based on this sentiment results on the tweets, we can predict that the BTC price will go up in the near future.

## ********************************
## BTC Price Prediction – RNN-LSTM

### The steps to modeling RNN-LSTM are as follows:
#### 1.	Gathering and visualizing BTC historical data.
#### 2.	Data pre-processing/preparation
   * Data splitting – Training and testing (encode the date, split data, process data)
   * Data scaling
   * Reshape the data for the model
#### 3.	Build the model – Train and test the model with the optimized parameters
#### 4.	Forecasting/Predicting the future prices

### Dataset overview
 * Our dataset comes from [Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD?p=BTC-USD). We downloaded and saved as an csv file. It covered historical data from 2014-11-20 to 2020-05-31. 
 * We loaded the data and converting it to a pandas dataframe.
 * The data is quite simple as it contains the date, open, high, low, close, adj_close, and volume.
 * We mainly used the “Close” price for the model.
 * Before building the model, we performed a data pre-processing.

### Data pre-processing/preparation
* Data splitting
  - Creating a 60-days prediction
  - Splitting into train and test set
 
* Data scaling
  - Scale the dataset because LSTM models are scale sensitive
  - We use the MinMaxScaler from scikit learn
 	 	 	
* Reshape the data
  - Reshape on the train data to give a new shape (without changing data) for the model.

### Build the model 
* To begin, we start with Sequential model. Our Sequential model will have one input layer and one output layer.
* The input layer in our network:
  - The first layer is the LSTM layer (to create our RNN)
  - The first argument is the number of nodes, which is set as 100.
  - Applies the ‘sigmoid’ activation function
  - Include the ‘input_shape’ for LSTM network. This is added so that the network knows what shape to expect.

* We used Dropout with a rate of 20% to combat overfitting.
* Our final layer is the densely connected NN output with one neuron and with ‘linear’ activation function.
* We compiled the model using ‘Adam’ (Adaptive moment estimation) as the optimizer and ‘Mean Squared Error’ as the loss function.
* Finally, we can do the fit/train to our RNN model.
  - The X_train and y_train are the variables (Close prices) we assigned to our BTC historical price data.
  - Epochs are the number of times the NN will train over the dataset. We use 50 epochs.
  - Batch size is the number of samples within the training set from the model. We use 10 batch_sizes.


### Predicting the future price
* After 50 epochs, the model is trained and can be used to forecast/predict future prices. We input the last 60 days of ‘Close’ prices in our model.predict() method to predict future prices for the next 60 days.
![](https://github.com/chrisnweb/Bitcoin-Price-Predictions/blob/master/Images/btcpredictiongraph.png)
 
### Result

#### The BTC price prediction results are modestly good. If they are wrong, then it is to be expected because not anyone and no machine can correctly predict the future.


## Conclusion

#### I have created reasonable simple models. CNN + LSTM and RNN + LSTM are great architectures that we can use to analyze and predict time-series information. The results for both models (tweets and price data) are better than I expected. Just imagine that we can use more input layers (more kinds of input), more arguments, more epochs, more batch size, and fine tune the parameters. Plus, I believe with more tweaking both models can be improved. Also, more checks and studies can be performed.

#### Predicting the future of cryptocurrencies are not easy. There are no experts or analysts that could make such predictions. Especially in the case with cryptocurrency like Bitcoin, I have found that the best analyst and price prediction expert is ‘you’.


## Questions to improve learning

#### *Are the prediction results good?*
##### Based on the BTC Price Prediction plot, it looks relatively good, does not look bad at all. Even though the patterns (actual BTC prices and predicted BTC prices) match fairly closely, the results are still apart from each other. Therefore, we need to further expand and develop the code to get better results.

#### *What can be done differently and continue to add to the project?*
##### 1.	Merge the 2 datasets: Historical price data and tweet data
##### 2.	Have more data (historical prices data from other crypto financial/exchange websites and much larger tweets)
##### 3.	There are many more time series models to learn and experiment. Beside the CNN, RNN and LSTM, run additional neural network models and carry out some statistical tests on the data.


## **CHEERIOS!**

## Reference and Resource

#### Irene Lui. 2018. Twitter-US-Airline-Sentiment_J2D_Project_Python. https://github.com/ireneliu521/Twitter-US-Airline-Sentiment_J2D_Project_Python/blob/master/Twitter%20US%20Airline%20Sentiment.ipynb. (2020).

#### Omer Berat Sezer. 2018. LSTM_RNN_Tutorials_with_Demo. https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo. (2020)

#### stackoverflow.com (2017). How to plot confusion matrix correctly. https://stackoverflow.com/questions/44189119/how-to-plot-confusion-matrix-correctly. Last accessed: 2020-06-19

#### Keras.io, Conv1D layer, https://keras.io/api/layers/convolution_layers/convolution1d/, June 14, 2020

#### scikit-learn.org, MinMaxScaler, https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html, June 14, 2020

#### twitter.com, #Bitcoin, https://twitter.com/search?q=%23Bitcoin, May 31, 2020

#### finance.yahoo.com, Bitcoin USD (BTC-USD), https://finance.yahoo.com/quote/BTC-USD?p=BTC-USD, May 31, 2020


```


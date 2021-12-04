# Next_word_Prediction
Next Word Prediction Model

Most of the keyboards in smartphones give next word prediction features; google also uses next word prediction based on our browsing history. So a preloaded data is also stored in the keyboard function of our smartphones to predict the next word correctly. In this article, I will train a Deep Learning model for next word prediction using Python. I will use the Tensorflow and Keras library in Python for next word prediction model.

For making a Next Word Prediction model, I will train a Recurrent Neural Network (RNN). 
## Next Word Prediction Model
**To start with our next word prediction model, let’s import some all the libraries we need for this task:**

![word1](https://user-images.githubusercontent.com/95492893/144710010-2f7aa1c9-cd7e-47c6-9bcb-dbbec3f535fc.PNG)

As I told earlier, Google uses our browsing history to make next word predictions, smartphones, and all the keyboards that are trained to predict the next word are trained using some data. So I will also use a dataset. You can download the dataset from 1661-0.txt file. 

**Now let’s load the data and have a quick look at what we are going to work with:**

![w2](https://user-images.githubusercontent.com/95492893/144710090-1ac71a24-6642-4788-8721-8ab470307eec.PNG)

**Now I will split the dataset into each word in order but without the presence of some special characters.**

![w3](https://user-images.githubusercontent.com/95492893/144710145-458e8fb1-af8d-4c04-8915-139ad082a785.PNG)

**Now the next process will be performing the feature engineering in our data. For this purpose, we will require a dictionary with each word in the data within the list of unique words as the key, and it’s significant portions as value.**

![image](https://user-images.githubusercontent.com/95492893/144710189-f478b70e-44f7-4fc6-accd-f65c2feec0e9.png)

## Feature Engineering
Feature Engineering means taking whatever information we have about our problem and turning it into numbers that we can use to build our feature matrix.

**Here I will define a Word length which will represent the number of previous words that will determine our next word. I will define prev words to keep five previous words and their corresponding next words in the list of next words.**

![w5](https://user-images.githubusercontent.com/95492893/144710271-a0bdcd44-6a9f-4bd9-9d28-68cbef48b404.PNG)

**Now I will create two numpy arrays x for storing the features and y for storing its corresponding label. I will iterate x and y if the word is available so that the corresponding position becomes 1.**
![w6](https://user-images.githubusercontent.com/95492893/144710341-169c4f58-30c5-4be7-9d1d-e4d0b050d7d5.PNG)

**Now before moving forward, have a look at a single sequence of words:**
![w7](https://user-images.githubusercontent.com/95492893/144710381-2abf0f08-b340-4aa9-aa5e-c6de50814b09.PNG)

## Building the Recurrent Neural network
As I stated earlier, I will use the Recurrent Neural networks for next word prediction model. Here I will use the LSTM model, which is a very powerful RNN.

![w8](https://user-images.githubusercontent.com/95492893/144710440-015044fb-f1d0-410d-a8f9-c0e7d5357cb7.PNG)

## Training the Next Word Prediction Model
I will be training the next word prediction model with 20 epochs:
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

**Now we have successfully trained our model, before moving forward to evaluating our model, it will be better to save this model for our future use.**

![w9](https://user-images.githubusercontent.com/95492893/144710540-79f126ee-3527-4fea-bb65-54fd7ba27283.PNG)

## Evaluating the Next Word Prediction Model

**Now let’s have a quick look at how our model is going to behave based on its accuracy and loss changes while training:**

![w10](https://user-images.githubusercontent.com/95492893/144710601-8b89cae7-e9c5-4088-931d-57f1eb921d8b.PNG)

![image](https://user-images.githubusercontent.com/95492893/144710716-0a77ad44-a111-4267-a408-efd2b2daf90e.png)

![w12](https://user-images.githubusercontent.com/95492893/144710763-8be98c30-8d44-4035-be1d-8dfa6f9caba7.PNG)

![image](https://user-images.githubusercontent.com/95492893/144710775-1df77641-b8c5-4924-a9ed-eb3f908ba87a.png)

## Testing Next Word Prediction Model

**Now let’s build a python program to predict the next word using our trained model. For this, I will define some essential functions that will be used in the process.**

![w13](https://user-images.githubusercontent.com/95492893/144710802-7e348395-74c9-4fc3-b63f-29dcc2ff2f47.PNG)

**Now before moving forward, let’s test the function, make sure you use a lower() function while giving input :**

![w14](https://user-images.githubusercontent.com/95492893/144710838-2906bfc9-dec1-4749-b7f7-3e9de919ad45.PNG)

**Note that the sequences should be 40 characters (not words) long so that we could easily fit it in a tensor of the shape (1, 40, 57). Not before moving forward, let’s check if the created function is working correctly.**

![w15](https://user-images.githubusercontent.com/95492893/144710883-914911f8-24a8-42f2-960f-eddffe4a56d7.PNG)

**Now I will create a function to return samples:**
![w16](https://user-images.githubusercontent.com/95492893/144710932-db463a2a-282c-4574-835f-e99984ef9e43.PNG)

**And now I will create a function for next word prediction:**
![w17](https://user-images.githubusercontent.com/95492893/144710980-f1b4109d-52b6-4d92-ac99-d9c9602afea7.PNG)

This function is created to predict the next word until space is generated. It will do this by iterating the input, which will ask our RNN model and extract instances from it. Now I will modify the above function to predict multiple characters:


def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]
   
   
   
 **Now I will use the sequence of 40 characters that we can use as a base for our predictions.**
 
 **quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]**


**Now finally, we can use the model to predict the next word:**

![w19](https://user-images.githubusercontent.com/95492893/144711153-d3af50c2-2f93-4e5e-9527-7fe3f42b654f.PNG)


























































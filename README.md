# Classifying Musical Genres using MFCCs

## Abstract

This project discusses the task of analyzing musical samples using Mel-frequency cepstrum coefficients (MFCCs) and deep neural networks.

## Introduction

I'm a huge music fan, with tastes ranging all the way from the Smiths to Yung Lean. Most of all, however, I was interested in music that spanned multiple genres: think Frank Ocean and Pink Floyd. A lot of this type of music lacks a label, so I was curious as to which genre(s) a machine learning model would categorize it as.

## The Dataset

The most difficult part of the project was definitely acquiring a usable dataset. Most popular songs are copyrighted (and therefore most datasets containing these were paid), so I was forced to get creative. I found a platform known as 7Digital and signed up as developers. We then scraped the platform for 30-second song previews until our dataset became fairly large at around 3GB.

My web scraper definitely has a lot of potential to be improved, and I can add many more genres as classifications. I am glad to have had the opportunity to learn some of the awesome applications of machine learning, and I hope that this project can serve as a springboard to help me explore my passions as a curious developer and aspiring data scientist.

## Genres Being Classified

The genres I will be classifying are:
1. Disco<br>
2. Hip-Hop<br>
3. Jazz<br>
4. Blues<br>
5. Classical<br>
6. Country<br>
7. Rock<br>
8. Reggae<br>
9. Pop<br>
10. Metal<br>

## Preprocessing

Having access to a large data set was a good first step, but to actually analyze each music sample, feeding in the entire raw file would not have sufficed, as each training sample needs to be standardized and numerical. After conducting extensive research into acoustics and sound analysis, I came across the industry standard for representing a sound sample as a vector - Mel-frequency cepstral coefficients (MFCCs). These are easily calculated using the publicly available Librosa Python library. 

MFCCs are calculated as follows:
1. Determine the Fourier transform of a given signal (sound sample).
2. Project the powers of the spectrum obtained above onto the mel scale using triangular overlapping windows.
3. Calculate the logs of the powers at each of the mel frequencies.
4. Take the discrete cosine transform of the mel log powers, treating it as a signal
5. The MFCCs are the amplitudes of the resulting spectrum.
 
## Deep Learning
 
After reading about the results that Tao and Sander (papers that I referred to) achieved, I decided to implement my model using a convolutional neural network (CNN). The CNN receives a 599-dimensional vector of Mel-frequency coefficients, each containing 128 frequencies which describe their window. The CNN consists of 3 hidden layers and finishes with a max pooling layer, a fully connected layer, and a softmax layer to end up with a 10-dimensional vector, one dimension for each genre I am trying to classify.
 
 Based on my analysis, I observed the following:

• Filter 250 picks up vocal thirds, i.e. multiple singers singing the same thing, but the notes are a major third (4 semitones) apart.
• Filter 14 seems to pick up vibrato singing.
• Filter 253 picks up various types of bass sounds and drum beats.
• Filter 242 picks up some kind of resounding ambience.

## Results

My convolutional neural network model is able to classify the genre of an unknown music sample (based on a 70-30 train/test split) correctly 64% of the time. While this isn't ideal in terms of accuracy, given the computing limitations of the machines I trained this model on and the limitations of creating feature-based representation for a song, this is still a pretty interesting and decent result. 

### Documentation

• downloader.py: 
This file iterates over all ‘.h5’ in the 7Digital directory and downloads a 30 seconds sample.

• mfcc.py: 
This file preprocesses each downloaded sound file by calculating MFCC for a 10-second window in the song and saving the result in a ‘.pp’ file.

• format_training_data.py: 
This file iterates over each of the ‘.pp’ files and generates two new files: ‘data’ and ‘labels’, each of which will be used as input to our convolutional neural network.

• model.py: 
This is the meat of my project; it generates the CNN using TensorFlow, feeding it with the previously created files ‘data’ and ‘labels’.  It will save the optimized model at the end as ‘model.pkt’.

## References

[1] Tao Feng, Deep learning for music genre classification, University of Illinois. https://courses.engr.illinois.edu/ece544na/fa2014/Tao_Feng.pdf
[2] Sander Dielman, Recommending Music on Spotify with Deep Learning, August 05, 2014. http://benanne.github.io/2014/08/05/spotify-cnns.html
[3] https://www.tensorflow.org

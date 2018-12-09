# Classifying Musical Genres using MFCCs

## Abstract

This project discusses the task of analyzing musical samples using Mel-frequency cepstrum coefficients (MFCCs) and deep neural networks.

## Introduction

Two out of the three of us spent the vast majority of this class completely lost. While we admire the concept of machine learning, its application came far more difficult than we had initially expected. Parth, the only one of us who had any previous experience with the subject, took it upon himself to walk the rest of the team through the concepts presented in class in relatable terms. He showed the rest of the team his previous projects, online examples, and guided us through the immense learnings he had gained from his internship at SymphonyAI over the past summer.

The three of us mutually agreed that we would skip the predefined project prompts and go with something we were all interested in. We eventually narrowed our project’s scope to two topics: either a recurrent neural network (RNN) that chooses a synonym for a word that the user would like to replace based on sentence content or use Mel Frequency Cepstral Coefficients (MFCCs) to classify the genre of a song using audio input. We finally chose the latter.
The three of us are all huge music fans, with tastes ranging all the way from the Smiths to Yung Lean. Most of all, however, we are interested in music that spans multiple genres: think Frank Ocean and Pink Floyd. A lot of this type of music lacks a label, so we were curious as to which genre(s) a machine learning model would categorize it as.

## The Dataset

The most difficult part of the project was definitely acquiring a usable dataset. Most popular songs are copyrighted (and therefore most datasets containing these were paid), so we were forced to get creative. We found a platform known as 7Digital and signed up as developers. We then scraped the platform for 30-second song previews until our dataset became fairly large at around 3GB.

From here, we plan to add to our model over Winter break. All three of us are proud of our work, and we will continue making Github commits to improve. Our web scraper has a lot of potential to be improved, and we can add many more genres as classifications. We are glad to have had the opportunity to learn the applications of machine learning, and we hope that the ML@B decal team likes our project as much as we do.

## Genres Being Classified

The genres we are classifying are:
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

Having access to a large data set was a good first step, but to actually analyze each music sample, feeding in the entire raw file would not have sufficed, as each training sample needs to be standardized and numerical. After conducting extensive research into acoustics and sound analysis, we came across the industry standard for representing a sound sample as a vector - Mel-frequency cepstral coefficients (MFCCs). These are easily calculated using the publicly available Librosa Python library. 

MFCCs are calculated as follows:
1. Determine the Fourier transform of a given signal (sound sample).
2. Project the powers of the spectrum obtained above onto the mel scale using triangular overlapping windows.
3. Calculate the logs of the powers at each of the mel frequencies.
4. Take the discrete cosine transform of the mel log powers, treating it as a signal
5. The MFCCs are the amplitudes of the resulting spectrum.
 
## The Graph
 
 After reading about the results that Tao and Sander (papers that we referred to) achieved, we decided to implement our model using a convolutional neural network (CNN). The CNN receives a 599 vector of mel frequency coefficients, each containing 128 frequencies which describe their window. The CNN consists of 3 hidden layers and finishes with a max pooling layer, a fully connected layer, and a softmax layer to end up with a 10-dimensional vector, one dimension for each genre we are trying to classify.
 
 Based on our analysis, we observed the following:

• Filter 250 picks up vocal thirds, i.e. multiple singers singing
  the same thing, but the notes are a major third (4 semitones) apart.
• Filter 14 seems to pick up vibrato singing.
• Filter 253 picks up various types of bass drum sounds.
• Filter 242 picks up some kind of ringing ambience.

## Results

Our convolutional neural network model is able to classify the genre of an unknown music sample (based on a 70-30 train/test split) correctly 47% of the time. While this isn't ideal in terms of accuracy, given the time we had for this project and the computing limitations of the machines we trained our model on, this is still a pretty interesting and decent result. 

### Documentation

• previewDownloader.py: 
USAGE: python previewDownloader.py [path to MSD data] 
This script iterate over all ‘.h5’ in a directory and download a 30 seconds sample from 7digital.

• preproccess.py: 
USAGE: python preproccess.py [path to MSD mp3 data] 
This script pre-processing the sound files. Calculating MFCC for a sliding window and saving the result in a ‘.pp’ file.

• formatInput.py: 
USAGE: python formatInput.py [path to MSD pp data] 
The script iterates over all ‘.pp’ files and generates ‘data’ and ‘labels’ that will be used as an input to the NN. 
Moreover, the script output a t-SNE graph at the end.

• train.py: 
USAGE: python train.py 
This script builds the neural network and feeds it with ‘data’ and ‘labels’.  When it is done it will save ‘model.final’.

### Complete Installation

<ul>
<li>Download the dataset files from https://www.dropbox.com/s/8ohx6m23co1qaz3/DataSet.zip?dl=0.</li>
<li>Unzip file</li>
<li>Place dataset files in the structure they are ordered in</li>
</ul>


## References

[1] Tao Feng, Deep learning for music genre classification, University of Illinois. https://courses.engr.illinois.edu/ece544na/fa2014/Tao_Feng.pdf
[2] Sander Dielman, Recommending Music on Spotify with Deep Learning, August 05, 2014. http://benanne.github.io/2014/08/05/spotify-cnns.html
[3] https://www.tensorflow.org

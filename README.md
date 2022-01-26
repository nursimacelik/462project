# Sentiment Analysis on IMDB User Reviews

In this project IMDB user reviews are classified into three classes: negative, neutral, positive. We apply various methods and compare the results. Combining Tf-idf with SVM results in the best score.

## Dataset

We use reviews from IMDB website. Training data set consists of 3000 samples whereas 750 samples are used for validation and testing.

## System

![Screenshot (4)](https://user-images.githubusercontent.com/33669453/151218588-710160b4-1815-427d-96a1-5174a500dc51.png)

## Results

![Screenshot (5)](https://user-images.githubusercontent.com/33669453/151220739-859087f2-db65-4619-9dc6-99eff288ddc2.png)

## Usage

1. Install necessary packages.

    `pip install -r requirements`

2. Train (There should be a training set in the directory "TRAIN" for this to work).

    `python3 train.py`

3. Test.

    `python3 462project_step2_Solis.py step2_model_Solis.pkl TEST `

     Outputs a string consisting of P,N,Z characters representing predicted class of each document.

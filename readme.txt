# BEFORE RUNNING

$ pip install -r requirements

# TRAINING

$ python train.py

Output: Performance metric values (accuracy, precision, recall, macro average).

All code for training resides in train.py. Models we tried are commented out, only the best one is uncommented. If you want to train a different model than the best one, make sure that the vectorizer (Count of Tf Idf) is set and model is uncommented.

# TESTING

$ python3 462project_step2_Solis.py step2_model_Solis.pkl TEST

Output: A string consisting of P,N,Z characters representing predicted class of each document.

# Kaggle-UBC-OCEAN
Competition submission for Kaggle UBC Ovarian Subtype Classification and Outlier Detection Competition


---------------------------------------------------------------------------------------------------------------------

## The Competition
The competition was a computer vision classification problem: presented with two types of images, the task was to classify within a set of various subtypes of ovarian cancer. For added complexity, an unkown fraction of the hidden test set are outliers - therefore a key part of the challenge was to write a "anomaly detection" routine.

## Our Work
We built upon an EfficientNet baseline using an AutoEncoder anomaly detection method. We aimed to classify anomalies based on larger reconstruction loss scores.

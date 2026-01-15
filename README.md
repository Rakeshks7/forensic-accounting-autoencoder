# Forensic Accounting Autoencoder 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

A "Top Tier" audit analytics tool that uses Deep Learning to detect fraud in General Ledger (GL) data. Unlike standard rules-based audits (e.g., "flag if > $5k"), this project uses an **Autoencoder with Categorical Embeddings** to learn the complex, non-linear patterns of normal business activity and flag highly suspicious anomalies.

##  Overview

Standard audit sampling misses 99% of transactions. This tool is designed to ingest 100% of a dataset and automatically identify the top 0.1% most anomalous entries without manual rule-writing.

It solves the "Needle in a Haystack" problem by training a neural network to compress and reconstruct valid transactions. Transactions that the network fails to reconstruct (high Reconstruction Error) are statistically likely to be fraud, errors, or rare events.

##  Key Features

* **Deep Autoencoder Architecture:** Compresses transaction data into a latent space and attempts to reconstruct it.
* **Categorical Embeddings:** Instead of basic One-Hot Encoding, this model learns dense vector representations for high-cardinality features (User IDs, Department Codes), capturing semantic relationships between users and departments.
* **Unsupervised Learning:** Requires **no labeled fraud data** to train. It learns "Normality" and flags deviations.
* **Log-Normal Simulation:** Includes a sophisticated data generator that mimics real-world financial data distributions (Benford's Law-like behavior) and injects specific fraud scenarios.

##  Tech Stack

* **Python 3.8+**
* **TensorFlow / Keras** (Model Architecture)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Preprocessing)
* **Matplotlib & Seaborn** (Visualization)

##  Disclaimer
For Educational and Research Purposes Only. This software is a proof-of-concept prototype. It is not intended to be used as a standalone tool for financial decisions, legal auditing, or actual fraud detection without significant customization and validation by qualified professionals. The authors assume no liability for the use of this code in production environments.
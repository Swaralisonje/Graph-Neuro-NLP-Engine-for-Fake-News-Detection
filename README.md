# Graph-Neuro-NLP-Engine-for-Fake-News-Detection

📌 Overview

This project is a Graph‑Neuro‑NLP Engine for detecting fake news by combining Graph Neural Networks (GNNs) and Natural Language Processing (NLP) techniques. It uses propagation graphs and textual semantics from social media datasets to train models that distinguish between real and fake news articles.

The engine includes:

Dataset preparation and preprocessing

Graph construction from social interactions

NLP embedding of news content

Model training and evaluation

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 Features

✔️ Build propagation graphs from social media datasets
✔️ NLP feature extraction from text content
✔️ Train & evaluate models using hybrid graph + text features
✔️ Support for Twitter15 and Twitter16 datasets
✔️ JSON metrics of performance ready for visualization

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🗃️ Dataset Preparation

The repository already includes the datasets/ folder. If using external datasets, prepare them in the same structure.

Supported datasets:

  Twitter15
  
  Twitter16

Each dataset should include graph propagation info and associated textual content.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔍 Model Output

After training and testing, the following files are generated:

  File                           Description
metrics_twitter15.json	  Evaluation metrics for Twitter15
metrics_twitter16.json	  Evaluation metrics for Twitter16
*.pth	                    Trained model weights

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 How It Works

Step 1: Preprocess Text

Tokenize and vectorize news text using NLP techniques.

Step 2: Construct Graphs

Use social propagation interactions to build graphs.

Step 3: Train Model

Feed combined features (text + graph) to the neural network.

Step 4: Evaluation

Evaluate using accuracy, F1, confusion matrix, etc.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 Visualization

Use the saved JSON metric files to plot performance curves (accuracy, loss, etc.) in tools like Matplotlib or Plotly.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📝 Acknowledgments

GNN and graph propagation research for fake news detection (e.g., UPFD, Twitter propagation graphs) 
GitHub

Inspiration from graph‑based fake news detection repositories and research

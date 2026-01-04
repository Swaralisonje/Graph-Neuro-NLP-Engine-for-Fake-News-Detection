# social_media_misinformation_detector.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from transformers import AutoTokenizer, AutoModel
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import json
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
import zipfile
import tarfile
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Social Media Misinformation Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #14171A;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fake-news {
        background-color: #FFCCCB;
        border-left: 5px solid #FF5252;
    }
    .true-news {
        background-color: #C8E6C9;
        border-left: 5px solid #4CAF50;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
    .probability-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 10px 0;
    }
    .probability-fill {
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        line-height: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🔍 Social Media Misinformation Detector</h1>', unsafe_allow_html=True)
st.markdown("""
This application uses a Graph Neural Network (GNN) to detect misinformation in social media posts. 
The model has been trained on Twitter15, Twitter16, and Pheme datasets.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a mode", 
                               ["Home", "Detect Misinformation", "Model Evaluation", "Train Model", "About"])

# Preprocessing functions
def preprocess_text(text):
    """Preprocess text for analysis"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Dataset loading functions
def load_twitter15():
    """Load Twitter15 dataset"""
    texts = []
    labels = []
    
    dataset_path = "datasets/twitter15"
    
    if not os.path.exists(dataset_path):
        st.error(f"Twitter15 dataset not found at {dataset_path}. Please download it first.")
        return [], []
    
    try:
        # Twitter15 typically has a structure with rumor and non-rumor categories
        for category in ['rumors', 'non-rumors']:
            category_path = os.path.join(dataset_path, category)
            if not os.path.exists(category_path):
                continue
                
            # Map category to label (1 for true, 0 for fake)
            label_val = 0 if category == 'rumors' else 1
            
            # Process each tweet file in the category
            for tweet_file in os.listdir(category_path):
                if tweet_file.endswith('.xml') or tweet_file.endswith('.json'):
                    file_path = os.path.join(category_path, tweet_file)
                    
                    try:
                        if tweet_file.endswith('.xml'):
                            # Parse XML file
                            tree = ET.parse(file_path)
                            root = tree.getroot()
                            
                            # Extract text - this will depend on the XML structure
                            # This is a generic example - adjust based on actual structure
                            for elem in root.iter('text'):
                                texts.append(elem.text)
                                labels.append(label_val)
                                
                        elif tweet_file.endswith('.json'):
                            # Parse JSON file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                tweet_data = json.load(f)
                                if 'text' in tweet_data:
                                    texts.append(tweet_data['text'])
                                    labels.append(label_val)
                    except Exception as e:
                        st.warning(f"Error processing {file_path}: {e}")
                        continue
        
        st.success(f"Loaded {len(texts)} samples from Twitter15 dataset")
        return texts, labels
    except Exception as e:
        st.error(f"Error loading Twitter15 dataset: {e}")
        return [], []

def load_twitter16():
    """Load Twitter16 dataset"""
    texts = []
    labels = []
    
    dataset_path = "datasets\twitter16"
    
    if not os.path.exists(dataset_path):
        st.error(f"Twitter16 dataset not found at {dataset_path}. Please download it first.")
        return [], []
    
    try:
        # Twitter16 has a similar structure to Twitter15
        for category in ['rumors', 'non-rumors']:
            category_path = os.path.join(dataset_path, category)
            if not os.path.exists(category_path):
                continue
                
            # Map category to label (1 for true, 0 for fake)
            label_val = 0 if category == 'rumors' else 1
            
            # Process each tweet file in the category
            for tweet_file in os.listdir(category_path):
                if tweet_file.endswith('.xml') or tweet_file.endswith('.json'):
                    file_path = os.path.join(category_path, tweet_file)
                    
                    try:
                        if tweet_file.endswith('.xml'):
                            # Parse XML file
                            tree = ET.parse(file_path)
                            root = tree.getroot()
                            
                            # Extract text - adjust based on actual structure
                            for elem in root.iter('text'):
                                if elem.text and elem.text.strip():
                                    texts.append(elem.text)
                                    labels.append(label_val)
                                
                        elif tweet_file.endswith('.json'):
                            # Parse JSON file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                tweet_data = json.load(f)
                                if 'text' in tweet_data and tweet_data['text'].strip():
                                    texts.append(tweet_data['text'])
                                    labels.append(label_val)
                    except Exception as e:
                        st.warning(f"Error processing {file_path}: {e}")
                        continue
        
        st.success(f"Loaded {len(texts)} samples from Twitter16 dataset")
        return texts, labels
    except Exception as e:
        st.error(f"Error loading Twitter16 dataset: {e}")
        return [], []


def load_phemescheme():
    """Load Pheme Rumour Scheme dataset"""
    texts = []
    labels = []

    dataset_path = "datasets/pheme-rumour-scheme-dataset"

    if not os.path.exists(dataset_path):
        st.error(f"Pheme Rumour Scheme dataset not found at {dataset_path}")
        return [], []

    try:
        for category in ['rumours', 'non-rumours']:
            category_path = os.path.join(dataset_path, category)
            if not os.path.exists(category_path):
                continue

            label_val = 0 if category == 'rumours' else 1

            for event in os.listdir(category_path):
                event_path = os.path.join(category_path, event)
                if not os.path.isdir(event_path):
                    continue

                # Source tweet
                source_tweet_path = os.path.join(event_path, 'source-tweet')
                if os.path.exists(source_tweet_path):
                    for tweet_file in os.listdir(source_tweet_path):
                        if tweet_file.endswith('.json'):
                            try:
                                with open(os.path.join(source_tweet_path, tweet_file), 'r', encoding='utf-8') as f:
                                    tweet_data = json.load(f)
                                    if 'text' in tweet_data and tweet_data['text'].strip():
                                        texts.append(tweet_data['text'])
                                        labels.append(label_val)
                            except Exception as e:
                                st.warning(f"Error reading {tweet_file}: {e}")
                                continue
        st.success(f"Loaded {len(texts)} samples from Pheme Rumour Scheme dataset")
        return texts, labels
    except Exception as e:
        st.error(f"Error loading Pheme Rumour Scheme dataset: {e}")
        return [], []

# Topic modeling class
class TopicModeler:
    def __init__(self, n_topics=8):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.topic_dictionaries = {}
        
    def fit(self, texts):
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Fit LDA model
        self.lda.fit(tfidf_matrix)
        
        # Create topic dictionaries
        self._create_topic_dictionaries()
        
        return self
    
    def _create_topic_dictionaries(self, n_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            self.topic_dictionaries[f"topic_{topic_idx}"] = top_words
            
    def transform(self, texts):
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        tfidf_matrix = self.vectorizer.transform(processed_texts)
        
        # Transform using LDA
        topic_distributions = self.lda.transform(tfidf_matrix)
        
        return topic_distributions
    
    def get_topic_words(self, n_words=10):
        return self.topic_dictionaries
    
    def get_document_topic_features(self, texts, topic_dict):
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Create feature vectors based on topic dictionaries
        features = []
        for text in processed_texts:
            text_features = []
            for topic, words in topic_dict.items():
                # Count how many topic words appear in the text
                count = sum(1 for word in words if word in text)
                text_features.append(count)
            features.append(text_features)
        
        return np.array(features)

# Graph construction functions
def build_topic_graphs(texts, topic_modeler, top_k=5):
    """Build topic graphs for the given texts"""
    graphs = []
    
    # Get topic distributions
    topic_distributions = topic_modeler.transform(texts)
    
    # Create a graph for each topic
    for topic_idx in range(topic_modeler.n_topics):
        # Create node features based on TF-IDF of topic words
        topic_words = topic_modeler.topic_dictionaries[f"topic_{topic_idx}"]
        vectorizer = TfidfVectorizer(vocabulary=topic_words)
        
        try:
            # Try to create TF-IDF features
            features = vectorizer.fit_transform(texts).toarray()
        except:
            # If no features can be created, use zeros
            features = np.zeros((len(texts), len(topic_words)))
        
        # Calculate cosine similarity between documents
        similarity_matrix = cosine_similarity(features)
        
        # Create edge list based on top-k similarities
        edge_list = []
        for i in range(len(texts)):
            # Get indices of top-k most similar documents
            similar_indices = similarity_matrix[i].argsort()[-top_k-1:-1][::-1]
            for j in similar_indices:
                if i != j and similarity_matrix[i, j] > 0:  # Only add edges with positive similarity
                    edge_list.append([i, j])
        
        # Convert to tensor format
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # Create self-loop if no edges
            edge_index = torch.tensor([[i for i in range(len(texts))], 
                                      [i for i in range(len(texts))]], dtype=torch.long)
        
        # Create graph data
        graph_data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index
        )
        
        graphs.append(graph_data)
    
    return graphs

# GNN Model Definition
class ATA_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_topics):
        super(ATA_GNN, self).__init__()
        self.num_topics = num_topics
        
        # GCN layers for each topic graph
        self.convs = nn.ModuleList()
        for _ in range(num_topics):
            self.convs.append(nn.ModuleList([
                GCNConv(input_dim, hidden_dim),
                GCNConv(hidden_dim, hidden_dim // 2)
            ]))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_topics * (hidden_dim // 2), 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x_list, edge_index_list, batch_list=None):
        topic_embeddings = []
        
        for i in range(self.num_topics):
            x, edge_index = x_list[i], edge_index_list[i]
            
            # First GCN layer
            x = self.convs[i][0](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            
            # Second GCN layer
            x = self.convs[i][1](x, edge_index)
            x = F.relu(x)
            
            # Global mean pooling if batch is provided
            if batch_list is not None:
                batch = batch_list[i]
                x = global_mean_pool(x, batch)
            else:
                # If no batch, just take mean of all nodes
                x = torch.mean(x, dim=0, keepdim=True)
                
            topic_embeddings.append(x)
        
        # Concatenate topic embeddings
        x = torch.cat(topic_embeddings, dim=1)
        
        # Classifier
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

# Training function
def train_model(dataset_name, num_epochs=100, num_topics=8):
    if dataset_name == "Twitter15":
        texts, labels = load_twitter15()
    elif dataset_name == "Twitter16":
        texts, labels = load_twitter16()
    elif dataset_name == "PhemeRumourScheme":
        texts, labels = load_phemescheme()
    else:
        st.error("Invalid dataset name")
        return None, None, None

    # Check if we have data
    if len(texts) == 0:
        st.error(f"No data loaded for {dataset_name}. Please check the dataset path and format.")
        return None, None, None
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Train topic modeler
    topic_modeler = TopicModeler(n_topics=num_topics)
    topic_modeler.fit(processed_texts)
    
    # Build topic graphs
    graphs = build_topic_graphs(processed_texts, topic_modeler)
    
    # Prepare data for training
    x_list = [graph.x for graph in graphs]
    edge_index_list = [graph.edge_index for graph in graphs]
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Initialize model
    input_dim = x_list[0].shape[1]  # Feature dimension
    hidden_dim = 64
    output_dim = 2  # Binary classification
    
    model = ATA_GNN(input_dim, hidden_dim, output_dim, num_topics)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    
    # Training loop
    model.train()
    losses = []
    accuracies = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(x_list, edge_index_list)
        
        # Calculate loss
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        accuracy = (pred == labels).float().mean()
        
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        
        # Update progress
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
    
    # Save model and topic modeler
    torch.save(model.state_dict(), f"ata_gnn_{dataset_name.lower()}.pth")
    with open(f"topic_modeler_{dataset_name.lower()}.pkl", "wb") as f:
        pickle.dump(topic_modeler, f)
    
    # Plot training results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Calculate final metrics
    model.eval()
    with torch.no_grad():
        output = model(x_list, edge_index_list)
        pred = output.argmax(dim=1)
        
        accuracy = accuracy_score(labels, pred)
        precision = precision_score(labels, pred, zero_division=0)
        recall = recall_score(labels, pred, zero_division=0)
        f1 = f1_score(labels, pred, zero_division=0)
        
        # For AUC, we need probability scores
        probas = torch.exp(output)[:, 1]  # Probability of class 1 (true)
        auc = roc_auc_score(labels, probas)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Precision", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")
    with col4:
        st.metric("F1 Score", f"{f1:.4f}")
    with col5:
        st.metric("AUC", f"{auc:.4f}")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    with open(f"metrics_{dataset_name.lower()}.json", "w") as f:
        json.dump(metrics, f)
    
    return model, topic_modeler, metrics

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find the main content of the page
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        # If no text found, try meta description
        if not text.strip():
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                text = meta_desc.get('content', '')
        
        return text[:1000]  # Limit to first 1000 characters
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return ""

# Function to create graph visualization
def create_graph_visualization(text, topic_words):
    # Create a simple graph for visualization
    G = nx.Graph()
    
    # Add nodes for the main text and each topic
    G.add_node("Input Text", size=500, color='lightblue')
    
    for topic, words in topic_words.items():
        G.add_node(topic, size=300, color='lightgreen')
        G.add_edge("Input Text", topic, weight=0.5)
        
        # Add words as nodes connected to their topic
        for word in words[:5]:  # Only show top 5 words per topic
            G.add_node(word, size=100, color='lightcoral')
            G.add_edge(topic, word, weight=0.3)
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.title("Text-Topic-Word Graph Representation")
    plt.axis('off')
    
    return plt

# Function to load a pre-trained model
def load_pretrained_model(dataset_name):
    """Load a pre-trained model and topic modeler"""
    model_path = f"ata_gnn_{dataset_name.lower()}.pth"
    topic_modeler_path = f"topic_modeler_{dataset_name.lower()}.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(topic_modeler_path):
        st.error(f"Pre-trained model for {dataset_name} not found. Please train the model first.")
        return None, None, None
    
    # Load topic modeler
    with open(topic_modeler_path, "rb") as f:
        topic_modeler = pickle.load(f)
    
    # Determine input dimension from topic modeler
    sample_text = "sample text"
    processed_text = preprocess_text(sample_text)
    topic_features = topic_modeler.get_document_topic_features([processed_text], topic_modeler.topic_dictionaries)
    input_dim = topic_features.shape[1]
    
    # Initialize model
    hidden_dim = 64
    output_dim = 2
    num_topics = topic_modeler.n_topics
    
    model = ATA_GNN(input_dim, hidden_dim, output_dim, num_topics)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load metrics if available
    metrics_path = f"metrics_{dataset_name.lower()}.json"
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    
    return model, topic_modeler, metrics

# Function to make predictions
def predict_with_model(model, topic_modeler, text):
    """Make a prediction using the trained model"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    if not processed_text.strip():
        return 0.5, 0.5, {}  # Return neutral prediction for empty text
    
    # Build topic graphs for the single text
    graphs = build_topic_graphs([processed_text], topic_modeler, top_k=1)
    
    # Prepare data for prediction
    x_list = [graph.x for graph in graphs]
    edge_index_list = [graph.edge_index for graph in graphs]
    
    # Make prediction
    with torch.no_grad():
        output = model(x_list, edge_index_list)
        probabilities = torch.exp(output)[0]  # Convert log probabilities to probabilities
        fake_prob = probabilities[0].item()
        true_prob = probabilities[1].item()
    
    # Get topic words
    topic_words = topic_modeler.get_topic_words()
    
    return fake_prob, true_prob, topic_words

# Home page
if app_mode == "Home":
    st.markdown("""
    ## Welcome to the Social Media Misinformation Detector
    
    This application uses an Advanced Text Analysis Graph Neural Network (ATA-GNN) to detect 
    misinformation in social media posts. The model has been trained on three benchmark datasets:
    
    - **Twitter15**: Contains tweets labeled as true or false rumors
    - **Twitter16**: Another Twitter dataset with rumor annotations
    - **Pheme**: A dataset of Twitter conversations around breaking news
    
    ### How it works:
    
    1. The system extracts text content from the provided social media URL
    2. It preprocesses the text (removing stopwords, lemmatization, etc.)
    3. The text is analyzed using topic modeling to identify key themes
    4. A Graph Neural Network processes the text through multiple topic graphs
    5. The model provides a prediction (True or Fake) with confidence probability
    
    ### Navigation:
    
    - **Detect Misinformation**: Analyze a social media post by providing its URL
    - **Model Evaluation**: View the performance metrics of our trained model
    - **Train Model**: Train the ATA-GNN model on the available datasets
    - **About**: Learn more about the technology behind this application
    
    ### Example URLs to try:
    
    - https://twitter.com/CNN/status/1311400137136058368
    - https://twitter.com/Reuters/status/1311668286688415744
    
    ### Setup Instructions:
    
    1. Download the Twitter15, Twitter16, and Pheme datasets
    2. Create a 'datasets' folder in your project directory
    3. Extract the datasets into subfolders: 'twitter15', 'twitter16', and 'pheme'
    4. Train the models using the 'Train Model' section
    5. Start detecting misinformation!
    """)

# Detection page
elif app_mode == "Detect Misinformation":
    st.markdown('<h2 class="sub-header">Detect Misinformation in Social Media Posts</h2>', unsafe_allow_html=True)
    
    # Dataset selection
    dataset_name = st.selectbox("Select dataset model to use:", 
                               ["Twitter15", "Twitter16", "PhemeRumourScheme"])
    
    # Load pre-trained model
    model, topic_modeler, metrics = load_pretrained_model(dataset_name)
    
    if model is not None and topic_modeler is not None:
        st.success(f"Loaded pre-trained model for {dataset_name}")
        
        # Input URL
        url = st.text_input("Enter the URL of a social media post:", 
                           placeholder="https://twitter.com/username/status/123456789")
        
        if st.button("Analyze Post") or url:
            if url:
                with st.spinner("Extracting content and analyzing..."):
                    # Extract text from URL
                    text_content = extract_text_from_url(url)
                    
                    if not text_content.strip():
                        st.error("Could not extract text content from this URL. Please try another URL.")
                    else:
                        # Display extracted text
                        st.subheader("Extracted Text Content")
                        st.text_area("", text_content, height=150)
                        
                        # Make prediction
                        fake_prob, true_prob, topic_words = predict_with_model(model, topic_modeler, text_content)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Create probability bars
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Probability Distribution")
                            
                            # Fake probability bar
                            st.markdown("**Fake News Probability**")
                            fake_percent = int(fake_prob * 100)
                            st.markdown(f"""
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: {fake_percent}%; background-color: {'#FF5252' if fake_prob > 0.5 else '#4CAF50'};">
                                    {fake_percent}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # True probability bar
                            st.markdown("**True News Probability**")
                            true_percent = int(true_prob * 100)
                            st.markdown(f"""
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: {true_percent}%; background-color: {'#4CAF50' if true_prob > 0.5 else '#FF5252'};">
                                    {true_percent}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### Prediction")
                            if fake_prob > 0.5:
                                st.markdown(f"""
                                <div class="result-box fake-news">
                                    <h2>🚫 Fake News</h2>
                                    <p>This content is likely misinformation with {fake_percent}% confidence.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-box true-news">
                                    <h2>✅ True News</h2>
                                    <p>This content is likely legitimate with {true_percent}% confidence.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Create and display graph visualization
                        st.subheader("Text-Topic Graph Representation")
                        fig = create_graph_visualization(text_content, topic_words)
                        st.pyplot(fig)
                        
                        # Display detected topics
                        st.subheader("Detected Topics and Keywords")
                        for topic, words in topic_words.items():
                            st.markdown(f"**{topic}**: {', '.join(words[:5])}")
            
            else:
                st.warning("Please enter a URL to analyze.")

# Model Evaluation page
elif app_mode == "Model Evaluation":
    st.markdown('<h2 class="sub-header">Model Performance Evaluation</h2>', unsafe_allow_html=True)
    
    # Dataset selection
    dataset_name = st.selectbox("Select dataset to evaluate:", 
                               ["Twitter15", "Twitter16", "PhemeRumourScheme"])
    
    # Load metrics
    metrics_path = f"metrics_{dataset_name.lower()}.json"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Accuracy</h3>
                <h2>{metrics.get('accuracy', 0) * 100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Precision</h3>
                <h2>{metrics.get('precision', 0) * 100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Recall</h3>
                <h2>{metrics.get('recall', 0) * 100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <h3>F1 Score</h3>
                <h2>{metrics.get('f1', 0) * 100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-box">
                <h3>AUC Score</h3>
                <h2>{metrics.get('auc', 0) * 100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset comparison
        st.subheader("Performance Across Datasets")
        
        # Try to load metrics for all datasets
        all_metrics = {}
        for ds in ["twitter15", "twitter16", "pheme-rumour-scheme-dataset"]:
            metrics_path = f"metrics_{ds}.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    all_metrics[ds] = json.load(f)
        
        if all_metrics:
            # Create comparison table
            comparison_data = []
            for ds, metrics in all_metrics.items():
                comparison_data.append({
                    'Dataset': ds,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1 Score': metrics.get('f1', 0),
                    'AUC': metrics.get('auc', 0)
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Visualization
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            # Bar chart for metrics comparison
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            x = np.arange(len(metrics_to_plot))
            width = 0.25
            
            for i, (ds, metrics) in enumerate(all_metrics.items()):
                values = [metrics.get(metric.lower(), 0) for metric in metrics_to_plot]
                ax[0].bar(x + (i-1)*width, values, width, label=ds)
            
            ax[0].set_xlabel('Metrics')
            ax[0].set_ylabel('Score')
            ax[0].set_title('Performance Metrics by Dataset')
            ax[0].set_xticks(x)
            ax[0].set_xticklabels(metrics_to_plot)
            ax[0].legend()
            ax[0].set_ylim(0, 1.0)
            
            # AUC comparison
            datasets = list(all_metrics.keys())
            auc_scores = [metrics.get('auc', 0) for metrics in all_metrics.values()]
            
            ax[1].bar(datasets, auc_scores)
            ax[1].set_xlabel('Dataset')
            ax[1].set_ylabel('AUC Score')
            ax[1].set_title('AUC Score by Dataset')
            ax[1].set_ylim(0, 1.0)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No comparison data available. Train models on other datasets to see comparisons.")
    else:
        st.warning(f"No evaluation metrics found for {dataset_name}. Please train the model first.")

# Train Model page
elif app_mode == "Train Model":
    st.markdown('<h2 class="sub-header">Train ATA-GNN Model</h2>', unsafe_allow_html=True)
    
    # Dataset selection
    dataset_name = st.selectbox("Select dataset to train on:", 
                               ["Twitter15", "Twitter16", "pheme-rumour-scheme-dataset"])
    
    # Check if dataset exists
    dataset_path = f"datasets/{dataset_name.lower()}"
    if not os.path.exists(dataset_path):
        st.error(f"Dataset not found at {dataset_path}. Please download the dataset first.")
        st.info("""
        Dataset download instructions:
        1. Create a 'datasets' folder in your project directory
        2. Download Twitter15, Twitter16, and Pheme datasets
        3. Extract each dataset into its own folder: 'twitter15', 'twitter16', 'pheme'
        4. Ensure the datasets contain the expected file structure
        """)
    else:
        # Training parameters
        col1, col2 = st.columns(2)
        
        with col1:
            num_epochs = st.slider("Number of epochs", min_value=10, max_value=500, value=100)
        
        with col2:
            num_topics = st.slider("Number of topics", min_value=4, max_value=16, value=8, step=4)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model, topic_modeler, metrics = train_model(dataset_name, num_epochs, num_topics)
                
                if model is not None:
                    st.success("Model training completed successfully!")

# About page
elif app_mode == "About":
    st.markdown('<h2 class="sub-header">About the Technology</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Advanced Text Analysis Graph Neural Network (ATA-GNN)
    
    This application uses a novel Graph Neural Network architecture specifically designed for 
    fake news detection in social media content. Unlike traditional approaches that rely on 
    user interactions or propagation patterns, our model focuses solely on textual content.
    
    #### Key Features:
    
    1. **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to identify key topics in the text
    2. **Multi-Graph Architecture**: Creates separate graphs for each detected topic
    3. **Graph Neural Networks**: Processes each topic graph using GCN layers
    4. **Feature Concatenation**: Combines embeddings from all topic graphs for final classification
    
    #### How it works:
    
    1. **Text Preprocessing**: Cleaning, tokenization, stopword removal, and lemmatization
    2. **Topic Modeling**: Identifying key themes and their representative words
    3. **Graph Construction**: Creating separate graphs for each topic with documents as nodes
    4. **GNN Processing**: Applying graph convolutional networks to each topic graph
    5. **Classification**: Combining outputs from all graphs to make a final prediction
    
    #### Datasets Used for Training:
    
    - **Twitter15**: 1,490 tweets labeled as true or false rumors
    - **Twitter16**: 1,282 tweets with similar annotation scheme
    - **Pheme**: 6,425 tweets from breaking news conversations
    
    #### Performance:
    
    Our model achieves state-of-the-art performance on all three datasets, outperforming 
    previous approaches like TextGCN, TensorGCN, and TCGNN.
    
    #### Applications:
    
    - Social media platforms for content moderation
    - News organizations for fact-checking
    - Educational institutions for media literacy training
    - Research purposes in computational journalism
    
    For more technical details, please refer to our publication:
    *Patel, A., & Sutrakar, V. K. (2024). Advanced Text Analytics - Graph Neural Network for Fake News Detection in Social Media.*
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Social Media Misinformation Detector | Powered by ATA-GNN</p>
        <p>For research purposes only</p>
    </div>
    """,
    unsafe_allow_html=True
)
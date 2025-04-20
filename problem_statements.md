Problem Statement:

Title: Sentiment Classification of User Input Using a Pretrained Transformer Model
In today's digital world, understanding user emotions and feedback from text data is crucial for applications such as customer support, product reviews, and social media monitoring. However, manually classifying sentiment from text is time-consuming and error-prone.

This project aims to build a simple and interactive web application that allows users to enter any sentence or phrase and receive an automatic sentiment analysis prediction—Positive or Negative.

To achieve this, we leverage a pretrained transformer-based model (DistilBERT) fine-tuned on the SST-2 dataset for binary sentiment classification. The application is built using Flask as the web framework and Hugging Face Transformers with PyTorch as the machine learning backend.
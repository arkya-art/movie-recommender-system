# 🎬 Movie Recommender System

A hybrid batch + real-time movie recommender built with **TensorFlow**, **Kafka**, **Flink**, and **Flask**. The project integrates offline Neural Collaborative Filtering (NCF) model training with real-time user interaction streaming using Apache Kafka and Apache Flink. Offline training ensures robust and accurate recommendations, while the streaming pipeline continuously captures user clicks and preferences, enabling the system to adapt and improve over time. A Flask-based REST API exposes endpoints for fetching recommendations and logging user feedback, and a React frontend provides a simple interface for users to interact with the system. The entire setup is containerized using Docker for reproducibility and easy deployment.
---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-green)
![Apache Kafka](https://img.shields.io/badge/Kafka-Event%20Streaming-black)
![Apache Flink](https://img.shields.io/badge/Flink-Stream%20Processing-pink)
![React](https://img.shields.io/badge/Frontend-React-blueviolet)


## 📑 Table of Contents

* [Introduction](#introduction)
* [System Architecture](#system-architecture)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Performance](#performance)
* [Future Work](#future-work)
* [Contributors](#contributors)

## 🏗️ System Architecture

![Recommender System Architecture](assets/system_architecture.png)

The system architecture is designed as a hybrid recommendation pipeline that integrates both offline training and real-time user feedback. A React.js web application acts as the front-end, enabling users to log in and request personalized movie recommendations. These requests are handled by a Flask-based Python API, which exposes two main endpoints — a GET request to fetch recommendations and a POST request to capture user clicks and feedback. The API leverages a TensorFlow-based Neural Collaborative Filtering (NCF) model, trained offline on the MovieLens dataset, to generate personalized predictions. To incorporate real-time behavior, all click interactions are published to a Kafka cluster (orchestrated with Docker and Zookeeper), ensuring reliable event streaming. These events are then consumed by Apache Flink, which processes the user interactions and stores them into a persistent database (in this implementation, CSV files). Before serving final recommendations, the NCF model checks against this stored feedback to avoid recommending already consumed items and to make more informed predictions. This architecture ensures a balance of robust offline accuracy and adaptive real-time learning for continuous improvement of recommendations.



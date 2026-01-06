
# Titanic Survival Prediction - KNN From Scratch

This repository contains a manual implementation of the K-Nearest Neighbors (KNN) algorithm, applied to the Titanic dataset. 
The goal of this project was to understand the underlying mechanics of the algorithm and data structures by building them in pure Python/NumPy..

## Key Implementations

* **Custom Min-Heap:** Implemented a binary heap data structure manually to manage the "Top-K" closest neighbors(efficient for online use in the future).
* **Weighted KNN:** Extended the base model using OOP (Inheritance) to support Inverse Distance Weighting (giving closer neighbors more influence).
* **Data Cleaning:** Manual normalization (Z-score) and handling of categorical features.

## Project Structure

* `min_heap.py`: The priority queue implementation.
* `knn.py`: The `KNNregressor` base class and the `KNNimproved` (weighted) subclass.
* `main.py`: Loads data, runs the training/testing loop, and plots the comparison graph.

## Results & Analysis

I compared the performance of the standard KNN against the Weighted KNN across different K values (1 to 25).

* **Best Accuracy:** about 80% was achieved with K=9.
* **Regular vs. Weighted:** there was neglegable chance surprisingly in favour of the regular KNN, but overall performed the same, maybe due to too much noise in the data

the result:
<img width="1728" height="1058" alt="Screenshot 2026-01-06 150928" src="https://github.com/user-attachments/assets/a24b4f5a-d34b-4a70-80ea-ce8813b69435" />


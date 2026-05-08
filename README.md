# Parallel Soft Clustering for Clinical Notes with OpenMPI
## IBA Karachi — Parallel and Distributed Computing, Spring 2026
### Project 21
Member || Role
Khansa Danish || Core Algorithm & Parallelisation
Ali Hamza || Clinical Text Processing & Feature Engineering
Zain Khan || Metrics, Testing & Validation
Arham Jumshaid || Data Distribution, Load Balancing & Visualisation

# What this project does
Electronic health records contain thousands of unstructured clinical notes. This project clusters them by medical similarity using Fuzzy C-Means (FCM) — a soft clustering algorithm where each note can partially belong to multiple clusters. Because FCM is computationally expensive at scale, the algorithm is parallelised across multiple processors using OpenMPI.
The full pipeline:

Raw clinical notes → numerical feature vectors (Python, NLP)
Parallel FCM clustering (C + OpenMPI)
Cluster validation, scaling analysis, and clinical interpretation (Python)

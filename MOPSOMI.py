import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import pdist, squareform
import random

# Load the leukemia dataset (this is a placeholder, replace with your actual dataset)
# data = pd.read_csv('leukemia_data.csv')
# X = data.drop('label', axis=1).values
# y = data['label'].values

X, y = make_classification(n_samples=100, n_features=500, n_informative=50, n_classes=2, random_state=42)


num_particles = 30
num_features = X.shape[1]
num_iterations = 50


def subset_size_determination():
    return random.randint(5, 50)

def gene_grouping(X):
    distance_matrix = squareform(pdist(X.T, 'correlation'))
    similar_groups = np.argsort(distance_matrix, axis=1)[:, :10]
    dissimilar_groups = np.argsort(distance_matrix, axis=1)[:, -10:]
    return similar_groups, dissimilar_groups

def particle_initialization():
    particles = np.random.randint(2, size=(num_particles, num_features))
    return particles

def particle_position_update(particle, velocity):
    updated_particle = particle + velocity
    updated_particle = np.clip(updated_particle, 0, 1)
    return updated_particle

def mutual_information_score(X, y):
    mi = mutual_info_classif(X, y)
    return mi

def calculate_objective(particles, X, y):
    scores = []
    mi = mutual_information_score(X, y)
    for particle in particles:
        selected_features = np.where(particle == 1)[0]
        if len(selected_features) == 0:
            scores.append(0)
        else:
            relevance = np.mean(mi[selected_features])
            redundancy = np.mean(np.corrcoef(X[:, selected_features].T))
            score = relevance - redundancy
            scores.append(score)
    return np.array(scores)


subset_size = subset_size_determination()
similar_groups, dissimilar_groups = gene_grouping(X)
particles = particle_initialization()
velocities = np.random.randn(num_particles, num_features)

for _ in range(num_iterations):
    for i in range(num_particles):
        particles[i] = particle_position_update(particles[i], velocities[i])
    scores = calculate_objective(particles, X, y)
    best_particle_idx = np.argmax(scores)
    best_particle = particles[best_particle_idx]


selected_genes = np.where(best_particle == 1)[0]
X_selected = X[:, selected_genes]


X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy, len(selected_genes)

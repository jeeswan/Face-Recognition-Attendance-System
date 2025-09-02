import numpy as np
from collections import Counter
import math

class CustomKNeighborsClassifier:
    """
    Custom K-Nearest Neighbors Classifier implemented from scratch.
    This implementation mimics the behavior of sklearn's KNeighborsClassifier.
    
    Features:
    - Euclidean distance calculation
    - K-nearest neighbor search
    - Majority voting for classification
    - Probability estimation
    - Compatible with sklearn's interface
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        """
        Initialize the Custom KNN Classifier.
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to use for classification
        weights : str, default='uniform'
            Weight function used in prediction. Only 'uniform' is implemented.
        metric : str, default='euclidean'
            Distance metric to use. Only 'euclidean' is implemented.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.n_samples_fit_ = None
        
    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
            
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        self.n_samples_fit_ = X.shape[0]
        
        return self
    
    def _calculate_distances(self, X):
        """
        Calculate Euclidean distances between X and all training samples.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        distances : array of shape (n_samples, n_samples_fit_)
            Distances between each pair of samples
        """
        X = np.asarray(X)
        distances = np.zeros((X.shape[0], self.X_train.shape[0]))
        
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                # Euclidean distance calculation
                distances[i, j] = np.sqrt(np.sum((x - x_train) ** 2))
                
        return distances
    
    def _get_neighbors(self, distances):
        """
        Get indices of k nearest neighbors for each sample.
        
        Parameters:
        -----------
        distances : array of shape (n_samples, n_samples_fit_)
            Distances between samples
            
        Returns:
        --------
        neighbors : array of shape (n_samples, n_neighbors)
            Indices of k nearest neighbors
        """
        neighbors = np.zeros((distances.shape[0], self.n_neighbors), dtype=int)
        
        for i in range(distances.shape[0]):
            # Get indices of k smallest distances
            neighbor_indices = np.argpartition(distances[i], self.n_neighbors)[:self.n_neighbors]
            neighbors[i] = neighbor_indices
            
        return neighbors
    
    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        y : array of shape (n_samples,)
            Predicted class labels
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Calculate distances
        distances = self._calculate_distances(X)
        
        # Get nearest neighbors
        neighbors = self._get_neighbors(distances)
        
        # Make predictions
        predictions = []
        for neighbor_indices in neighbors:
            # Get labels of k nearest neighbors
            neighbor_labels = self.y_train[neighbor_indices]
            
            # Majority voting
            label_counts = Counter(neighbor_labels)
            predicted_label = max(label_counts, key=label_counts.get)
            predictions.append(predicted_label)
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the provided data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        probabilities : array of shape (n_samples, n_classes)
            Class probabilities
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Calculate distances
        distances = self._calculate_distances(X)
        
        # Get nearest neighbors
        neighbors = self._get_neighbors(distances)
        
        # Calculate probabilities
        probabilities = []
        for neighbor_indices in neighbors:
            # Get labels of k nearest neighbors
            neighbor_labels = self.y_train[neighbor_indices]
            
            # Count occurrences of each class
            label_counts = Counter(neighbor_labels)
            
            # Calculate probabilities
            proba = np.zeros(len(self.classes_))
            for i, class_label in enumerate(self.classes_):
                proba[i] = label_counts.get(class_label, 0) / self.n_neighbors
                
            probabilities.append(proba)
            
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels
            
        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns:
        --------
        params : dict
            Parameter names mapped to their values
        """
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **params : dict
            Estimator parameters
            
        Returns:
        --------
        self : object
            Estimator instance
        """
        for key, value in params.items():
            if key in ['n_neighbors', 'weights', 'metric']:
                setattr(self, key, value)
        return self

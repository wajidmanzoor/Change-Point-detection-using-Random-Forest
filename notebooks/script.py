import numpy as np

def changeforest(Observations, n, p_hat, delta):
    def BinarySegmentation(xi, u, v, p, delta):
        if v - u < 2 * delta * n:
            return set()
        
        s, likelihood_ratios = TwoStepSearch(xi, u, v, p, delta)
        q = ModelSelection(likelihood_ratios, delta)
        
        if q < 0.02:
            alpha_left = BinarySegmentation(xi, u, s, p, delta)
            alpha_right = BinarySegmentation(xi, s+1, v, p, delta)
            return alpha_left.union({s}).union(alpha_right)
        else:
            return set()
    
    def TwoStepSearch(xi, u, v, p_hat, delta):
        def log_eta(x):
            eta = np.exp(-6)
            return np.log((1 - eta) * x + eta)
        
        def train_classifier(xi, u, s, v):
            pi = (s - u - 1) // (v - u - 1)
            classifiers = []
            for j in range(1, 4):
                X_train = xi[u:v+1]
                y_train = np.concatenate([np.ones(s - u + 1), np.zeros(v - s)])
                classifier = train_binary_classifier(X_train, y_train)
                classifiers.append(classifier)
            return classifiers
        
        def train_binary_classifier(X_train, y_train):
            # Placeholder for training binary classifier
            return None
        
        def compute_likelihood_ratios(classifiers, xi, u, s, v):
            likelihood_ratios = np.zeros((v - u + 1, 2, 3))
            for i in range(u, v + 1):
                for k in range(2):
                    for j in range(3):
                        likelihood_ratios[i - u, k, j] = log_eta(classifiers[j].predict_proba(xi[i])[0][k])
            return likelihood_ratios
        
        classifiers = train_classifier(Observations, u, s, v)
        likelihood_ratios = compute_likelihood_ratios(classifiers, Observations, u, s, v)
        
        max_sum = float('-inf')
        s_hat = None
        for s in range(u + delta * n, v - delta * n + 1):
            current_sum = 0
            for i in range(u, s + 1):
                current_sum += np.sum(likelihood_ratios[i - u, 0, :])
            for i in range(s + 1, v + 1):
                current_sum += np.sum(likelihood_ratios[i - u, 1, :])
            if current_sum > max_sum:
                max_sum = current_sum
                s_hat = s
        
        return s_hat, likelihood_ratios
    
    def ModelSelection(likelihood_ratios, delta):
        def compute_maximal_gain(likelihood_ratios, u, v):
            max_gain = float('-inf')
            for s in range(u + delta * n + 1, v - delta * n):
                for j in range(3):
                    current_sum = np.sum(likelihood_ratios[u:v+1, 0, j]) + np.sum(likelihood_ratios[u:v+1, 1, j])
                    max_gain = max(max_gain, current_sum)
            return max_gain
        
        def compute_p_value(likelihood_ratios, max_gain, delta):
            num_permutations = 200
            greater_than_max_gain = 0
            for _ in range(num_permutations):
                permuted_indices = np.random.permutation(len(likelihood_ratios))
                permuted_likelihood_ratios = likelihood_ratios[permuted_indices]
                gain = compute_maximal_gain(permuted_likelihood_ratios, u, v)
                if gain > max_gain:
                    greater_than_max_gain += 1
            return greater_than_max_gain / num_permutations
        
        max_gain = compute_maximal_gain(likelihood_ratios, u, v)
        p_value = compute_p_value(likelihood_ratios, max_gain, delta)
        return p_value
    
    return BinarySegmentation(Observations, 0, n-1, p_hat, delta)

# Example usage
Observations = np.random.randn(100)
n = len(Observations)
p_hat = None  # Placeholder for classifier
delta = 0.1
change_points = changeforest(Observations, n, p_hat, delta)
print("Change points:", change_points)

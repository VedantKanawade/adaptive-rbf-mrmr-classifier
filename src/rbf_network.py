import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# RBF TRANSFORM

def rbf_transform(X, centers, sigma):

    # Compute pairwise distances: (N, M)
    distances = cdist(X, centers, metric='euclidean')
    
    # Apply Gaussian RBF
    Phi = np.exp(- distances**2 / (2 * sigma**2))
    
    return Phi

# MAHALANOBIS J SCORE

def compute_J_scores(Phi, y):

    N, M = Phi.shape
    
    # Separate by class
    mask_pos = (y ==  1)
    mask_neg = (y == -1)
    
    N_pos = np.sum(mask_pos)
    N_neg = np.sum(mask_neg)
    
    J_scores = np.zeros(M)
    
    for j in range(M):
        phi_j = Phi[:, j]
        
        # Overall mean
        mu = np.mean(phi_j)
        
        # Class means
        mu_pos = np.mean(phi_j[mask_pos])
        mu_neg = np.mean(phi_j[mask_neg])
        
        # Between-class scatter
        SB = N_pos * (mu_pos - mu)**2 + N_neg * (mu_neg - mu)**2
        
        # Within-class scatter
        SW = np.sum((phi_j[mask_pos] - mu_pos)**2) + \
             np.sum((phi_j[mask_neg] - mu_neg)**2)
        
        # Avoid division by zero
        if SW < 1e-10:
            J_scores[j] = 0
        else:
            J_scores[j] = SB / SW
    
    return J_scores

# mRMR SELECTION WITH CV STOPPING

def mrmr_selection_cv(Phi, y, J_scores, n_folds=5, max_features=None):

    N, M = Phi.shape
    
    if max_features is None:
        max_features = min(M, N // 3)  # reasonable upper limit
    
    selected = []
    candidates = list(range(M))
    cv_history = []
    consecutive_drops = 0
    best_cv = 0
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    while len(selected) < max_features and len(candidates) > 0:
        
        if len(selected) == 0:
            # Round 1: pick highest J score
            mrmr_scores = J_scores.copy()
        else:
            # Compute mRMR scores for all candidates
            Phi_selected = Phi[:, selected]
            
            mrmr_scores = np.full(M, -np.inf)
            
            for idx in candidates:
                phi_candidate = Phi[:, idx].reshape(-1, 1)
                
                # Compute correlation with all selected features
                correlations = []
                for s in selected:
                    phi_s = Phi[:, s]
                    corr = np.corrcoef(phi_candidate.flatten(), phi_s)[0, 1]
                    correlations.append(abs(corr))
                
                # mRMR = relevance - redundancy
                redundancy = np.mean(correlations)
                mrmr_scores[idx] = J_scores[idx] - redundancy
        
        # Pick best mRMR score from candidates
        best_idx = max(candidates, key=lambda i: mrmr_scores[i])
        selected.append(best_idx)
        candidates.remove(best_idx)
        
        # Evaluate with CV
        Phi_subset = Phi[:, selected]
        cv_acc = cross_validate(Phi_subset, y, kf)
        cv_history.append(cv_acc)
        
        # Check stopping criterion
        if cv_acc > best_cv:
            best_cv = cv_acc
            consecutive_drops = 0
        else:
            consecutive_drops += 1
        
        # Stop if 2 consecutive drops
        if consecutive_drops >= 2:
            # Remove last 2 features that caused drops
            selected = selected[:-2]
            cv_history = cv_history[:-2]
            break
    
    return selected, cv_history


def cross_validate(Phi, y, kf):

    accuracies = []
    
    for train_idx, val_idx in kf.split(Phi):
        X_train, X_val = Phi[train_idx], Phi[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train Ridge classifier
        clf = Ridge(alpha=0.1)
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_val)
        y_pred = np.sign(y_pred)  # Convert to +1/-1
        
        # Accuracy
        acc = np.mean(y_pred == y_val)
        accuracies.append(acc)
    
    return np.mean(accuracies)

# COMPLETE EVALUATION PIPELINE

def evaluate_sigma(sigma, X, y, verbose=False):

    if verbose:
        print(f"Evaluating σ = {sigma:.4f}")
    
    # Step 1: RBF Transform (all samples as centers)
    Phi = rbf_transform(X, X, sigma)
    
    if verbose:
        print(f"  RBF transform: {X.shape} → {Phi.shape}")
    
    # Step 2: Compute J scores
    J_scores = compute_J_scores(Phi, y)
    
    if verbose:
        print(f"  J scores computed: max={J_scores.max():.4f}, min={J_scores.min():.4f}")
    
    # Step 3: mRMR selection with CV stopping
    selected_idx, cv_history = mrmr_selection_cv(Phi, y, J_scores)
    
    H = len(selected_idx)
    cv_acc = cv_history[-1] if cv_history else 0
    
    if verbose:
        print(f"  Selected {H} neurons, CV accuracy = {cv_acc:.4f}")
    
    return cv_acc, H, selected_idx

def refine_sigma_per_neuron(X, selected_idx, H):
    """
    Compute per-neuron sigma using P-nearest neighbor rule.
    
    Parameters:
        X            : (N, d) training data
        selected_idx : list of selected neuron indices
        H            : number of selected neurons
    
    Returns:
        sigma_refined : (H,) array of per-neuron sigmas
    """
    # Extract selected centers
    selected_centers = X[selected_idx]
    
    # P = number of neighbors to consider
    P = max(1, H // 5)
    
    # Compute pairwise distances between selected centers
    D_centers = cdist(selected_centers, selected_centers, metric='euclidean')
    
    sigma_refined = np.zeros(H)
    
    for j in range(H):
        # Distances from center j to all others
        distances = D_centers[j, :].copy()
        
        # Exclude self
        distances[j] = np.inf
        
        # Find P nearest neighbors
        nearest_P = np.partition(distances, min(P-1, H-2))[:P]
        
        # Sigma = mean distance to P nearest neighbors
        sigma_refined[j] = np.mean(nearest_P)
    
    return sigma_refined

def train_output_weights(X, y, selected_idx, sigma_refined):
    """
    Train output weights using least squares.
    
    Parameters:
        X             : (N, d) training data
        y             : (N,) labels
        selected_idx  : list of selected neuron indices
        sigma_refined : (H,) per-neuron sigmas
    
    Returns:
        W        : (H,) output weights
        Phi      : (N, H) final RBF feature matrix
        accuracy : training accuracy
    """
    N = X.shape[0]
    H = len(selected_idx)
    
    # Recompute Φ with refined sigmas
    Phi = np.zeros((N, H))
    
    for j in range(H):
        center_j = X[selected_idx[j]]
        sigma_j = sigma_refined[j]
        
        distances = np.linalg.norm(X - center_j, axis=1)
        Phi[:, j] = np.exp(- distances**2 / (2 * sigma_j**2))
    
    # Solve for W with regularization
    Phi_T = Phi.T
    Phi_T_Phi = Phi_T @ Phi
    Phi_T_y = Phi_T @ y
    
    regularization = 1e-6
    Phi_T_Phi_reg = Phi_T_Phi + regularization * np.eye(H)
    
    W = np.linalg.solve(Phi_T_Phi_reg, Phi_T_y)
    
    # Compute training accuracy
    y_pred = np.sign(Phi @ W)
    y_pred[y_pred == 0] = 1
    accuracy = np.mean(y_pred == y)
    
    return W, Phi, accuracy
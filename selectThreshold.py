import numpy as np


def select_threshold(yval, pval):
    # Initialize the best F1 score and the best epsilon value
    f1 = 0
    best_eps = 0
    best_f1 = 0

    # Iterate through a range of potential epsilon values 
    for epsilon in np.linspace(np.min(pval), np.max(pval), num=1001):
        
        # Make prediction whether each example is an anomaly (True if pval < epsilon)
        predictions = np.less(pval, epsilon)

        # Calculate True Positives (tp), False positives (fp), and False Negatives (fn)
        tp = np.sum(np.logical_and(predictions, yval))
        fp = np.sum(np.logical_and(predictions, yval ==0))
        fn = np.sum(np.logical_and(np.logical_not(predictions), yval == 1))

        # Calculate precision and recall
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        # Calculate F1 score
        f1 = (2 * precision * recall) / (precision + recall)

        # Update best F1 and epsilon if current F1 score is better
        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon

    return best_eps, best_f1

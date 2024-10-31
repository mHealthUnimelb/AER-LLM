import numpy as np


def KL(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Avoid neither a nor b is equal to 0. 
    epsilon = 1e-8

    b = b + epsilon

    KL_value = np.sum(np.where(a != 0, a * np.log(a / b), 0))
    return round(KL_value, 4)

def BC(a,b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Calculate the Bhattacharyya coefficient
    epsilon = 1e-10
    b_coefficient = round(np.sum(np.sqrt(a * b)),4)

    b_distance = round(-np.log(b_coefficient+epsilon),4)
    return b_coefficient

def R(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    epsilon = 1e-10
    covariance = np.cov(a,b, ddof=0)[0][1]

    var_a = np.var(a, ddof=0)
    var_b = np.var(b, ddof=0)
    R_square = round(covariance**2 / (var_a * var_b + epsilon), 4)
    return R_square

def cal_entropy(emo_probs):
    emo_probs = np.array(emo_probs)
    non_0_probs = emo_probs[emo_probs > 0]
    entropy = abs(np.sum(non_0_probs*np.log2(non_0_probs)))
    return round(entropy,4)


def js_distance(p, q):
    # Calculate M as the average of P and Q
    p= np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    m = 0.5 * (p + q)
    
    # Calculate JS Divergence

    js_divergence = 0.5 * KL(p, m) + 0.5 * KL(q, m)

    return np.sqrt(js_divergence)


def ECE(samples, trues, n_bims = 5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, n_bims + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)
    true_labels = np.argmax(trues, axis=1)
    
    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece
import pandas as pd
import json
import eval_metrics as em
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score


def get_results(folder_path):
    pred_data = pd.read_csv(f'./{folder_path}/pred.csv', names=["neutral", "happy", "angry", "sad"])
    truth_data = pd.read_csv(f'./{folder_path}/truth.csv', names=["neutral", "happy", "angry", "sad"])
    log = json.load(open(f'./{folder_path}/log.json'))
    pred = pred_data.values.tolist()
    truth = truth_data.values.tolist()
    return pred, truth, log

def get_eval(pred, truth):
    KL_dist = []
    R_square = []
    BC = []
    JS_distance = []
    for i in range(len(pred)):
        KL_dist.append(round(em.KL(pred[i], truth[i]),6))
        b_coefficient = em.BC(pred[i], truth[i])
        BC.append(round(b_coefficient,6))
        R_square.append(round(em.R(pred[i], truth[i]),6))
        JS_distance.append(round(em.js_distance(pred[i], truth[i]),6))

    eval_metrics = pd.DataFrame({
        'ground_truth': truth,
        'prediction': pred,
        'KL': KL_dist,
        'BC': BC,
        'R_square': R_square,
        'JS': JS_distance
    })
    return eval_metrics

def cal_entropy(emo_probs):
    emo_probs = np.array(emo_probs)
    non_0_probs = emo_probs[emo_probs > 0]
    entropy = abs(np.sum(non_0_probs*np.log2(non_0_probs)))
    return round(entropy,4)

def add_entropy(eval_metrics):
    entropy = []
    for i in range(len(eval_metrics)):
        entropy.append(cal_entropy(eval_metrics.iloc[i,0]))
    eval_metrics['entropy'] = entropy
    return eval_metrics

def whole_eval(path):
    path = path
    pred, truth, log = get_results(path)
    eval_metric = get_eval(pred, truth)
    # add entropy column
    eval_metric = add_entropy(eval_metric)
    entropy_vs_evalmetrics(eval_metric)
    print("Mean of JS", round(eval_metric['JS'].mean(),2))
    print("Mean of BC score: ", round(eval_metric['BC'].mean(),2))
    print("Mean of R square: ", round(eval_metric['R_square'].mean(),2))
    pred = np.array(pred)
    truth = np.array(truth)
    maj_pred_prob, maj_truth_prob, maj_pred_label, maj_truth_label = majority_agree(pred, truth)
    ece = ECE(pred, truth)
    acc = accuracy_score(maj_truth_label, maj_pred_label)
    print("ECE: ", round(ece[0]*100,2))
    print("UAC: ", round(acc*100,2))
    w_f1 = f1_score(maj_truth_label, maj_pred_label, average='weighted')
    print("Weighted F1: ", round(w_f1*100,2))
    return eval_metric

def cal_vote_by_entropy(d):
    d['true'] = d['ground_truth'].apply(lambda x: np.argmax(x))
    d['pred'] = d['prediction'].apply(lambda x: np.argmax(x))
    d = d[d['entropy'].apply(lambda x:x!= 1.5835)]

    print("overall acc: ", round((d['true'] == d['pred']).mean()*100,2))
    print("overall W-F1: ", round(f1_score(d['true'], d['pred'], average='weighted')*100,2))
    d_e = d.groupby('entropy')

    acc_by_entropy = d_e.apply(lambda x: round((x['true'] == x['pred']).mean()*100,2))
    acc_by_entropy.columns = ['entropy', 'accuracy']
    print("entropy = 0.0: acc = ", acc_by_entropy[0])
    print("entropy = 0.9149: acc = ", acc_by_entropy[0.9149])
    
    f1_by_group = d_e.apply(lambda x: round(f1_score(x['true'], x['pred'], average='weighted')*100,2))
    print("entropy = 0.0: W-F1 = :", f1_by_group[0])
    print("entropy = 0.9149: W-F1 = ", f1_by_group[0.9149])

__name__ == '__main__':
    eval_df = whole_eval("prediction/0825_context30")
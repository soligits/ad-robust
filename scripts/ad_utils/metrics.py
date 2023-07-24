from sklearn.metrics import roc_auc_score, average_precision_score

def AUROC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def AUPR(y_true, y_pred):
    return average_precision_score(y_true, y_pred)


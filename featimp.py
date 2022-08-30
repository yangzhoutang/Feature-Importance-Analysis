from rfpimp import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import shap


def featimp_cor(X, y):
    '''
    feature importance by Spearman's rank correlation coefficient
    '''
    n_feat = X.shape[1]
    imp = np.zeros(n_feat)
    for i in range(n_feat):
        imp[i] = np.abs(spearmanr(X.iloc[:, i], y).correlation)
    
    I = pd.DataFrame({'Feature': X.columns, 'Importance': imp})
    I = I.sort_values('Importance', ascending=False)
    I = I.set_index('Feature')

    return I

def featimp_pca(X):
    '''
    feature importance by principle component analysis
    '''
    pca = PCA(n_components=1)
    pca.fit(X)

    # use the first component as feature importance
    imp = np.abs(pca.components_[0])
    I = pd.DataFrame({'Feature': X.columns, 'Importance': imp})
    I = I.sort_values('Importance', ascending=False)
    I = I.set_index('Feature')

    # also return percentage of the variance covered by the first component
    evr = pca.explained_variance_ratio_[0]

    return I, evr

def featimp_mrmr(X, y):
    '''
    feature importance by minimal-redundancy-maximal-relevance(mRMR)
    '''
    n_feat = X.shape[1]
    selected, imp = [], []
    remained = list(range(n_feat))
    
    # use Spearman's rank correlation coefficient
    cor = spearmanr(pd.concat([X, y], axis=1)).correlation
    cor = np.abs(cor)
    idx = cor[n_feat, remained].argmax()
    selected.append(idx)
    remained.remove(idx)
    imp.append(cor[n_feat, idx])

    for i in range(n_feat - 1):
        Imrmr = cor[n_feat, remained] - cor[selected].T[remained].T.mean(axis=0)
        idx = remained[Imrmr.argmax()]
        selected.append(idx)
        remained.remove(idx)
        imp.append(Imrmr.max())

    I = pd.DataFrame({'Feature': X.columns[selected], 'Importance': imp})
    I = I.set_index('Feature')
    return I

def compare_strategies(X, y, I_cor, I_pca, I_mrmr, I_perm, I_drop):

    # train random forest model
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, oob_score=True)
    rf.fit(X, y)
    final_score = rf.oob_score_

    ks = list(range(1, 6))

    # shap values
    explainer = shap.Explainer(rf)
    shap_values = explainer(X)
    I = shap_values.base_values.mean(axis=0)

    # accumulate importance for shap
    Iaccu = np.zeros(X.shape[1])
    Iaccu[0] = I[0]
    for i in range(1, len(I)):
        Iaccu[i] = Iaccu[i-1] + I[i]
    Iaccu[3] = (1 + Iaccu[2]) / 2
    Iaccu[4] = 1
    # Interpolation
    Iaccu = (Iaccu - Iaccu[0]) * (final_score - Iaccu[0]) + Iaccu[0]
    plt.plot(list(range(1, 6)), Iaccu)

    # get scores for top k features
    methods = ['Spearman', 'PCA', 'mRMR', 'permutation', 'drop column']
    Is = [I_cor, I_pca, I_mrmr, I_perm, I_drop]
    for method, I in zip(methods, Is):
        scores = []
        for k in range(1, 5):
            curX = X[I.index[:k]]
            score = RandomForestClassifier(n_estimators=100, 
                            min_samples_leaf=10, 
                            oob_score=True).fit(curX, y).oob_score_
            scores.append(score)
        scores.append(final_score)

        # plot
        plt.plot(ks, scores, label=method)

    plt.title("Rent interest level")
    plt.xlabel("Top k most important features")
    plt.ylabel("OOB score")
    plt.legend()
    plt.show()

def select_feature(X, y, featimp):

    n_feat = X.shape[1]
    baseline_socre = 0
    features = list(X.columns)
    X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.15)

    scores = []
    dropped_features = []
    for i in range(n_feat):
        rf = RandomForestClassifier(n_estimators=100, 
                                min_samples_leaf=10, 
                                oob_score=True).fit(X_train[features], y_train)
        score = rf.oob_score_
        if score <= baseline_socre:
            break

        baseline_socre = score
        scores.append(score)

        # find least important feature
        if featimp == 'Spearman':
            I = featimp_cor(X[features], y)
        elif featimp == 'PCA':
            I, _ = featimp_pca(X[features])
        elif featimp == 'mRMR':
            I = featimp_mrmr(X[features], y)
        elif featimp == 'permutation':
            I = permutation_importances(rf, X_test[features], y_test, oob_classifier_accuracy)
        elif featimp == 'drop column':
            I = dropcol_importances(rf, X_train[features], y_train, X_test[features], y_test, 
                                oob_classifier_accuracy)

        features.remove(I.index[-1])
        dropped_features.append(I.index[-1])

    return scores, dropped_features

def get_var(X, y, N):
    I = pd.DataFrame(np.zeros((N, X.shape[1])), columns=X.columns)
    for i in range(N):
        idx = np.random.choice(np.arange(0, X.shape[0]), X.shape[0], replace=True)
        curX, cury = X.iloc[idx, ], y[idx]
        I.iloc[i, :] = featimp_cor(curX, cury)['Importance']
    
    mean = I.mean(axis=0)
    var = I.std(axis=0)
    I = pd.DataFrame({'Feature': X.columns[::-1], 'Importance': mean, 'Variance': var})
    I = I.sort_values('Importance', ascending=False)
    I = I.set_index('Feature')
    plot_importances(I, imp_range=(-.025, 0.3))

    # error plot
    unit = 1
    ypadding = 0.1
    yloc = []
    y = unit / 2 + ypadding
    yloc.append(y)
    for i in range(1, X.shape[1]):
        y += unit + ypadding
        yloc.append(y)
    
    plt.errorbar(I['Importance'], yloc, xerr=3*I['Variance'])

def get_pvalue(X, y, N=80):

    I = featimp_cor(X, y)
    null_dist = pd.DataFrame(np.zeros((N, X.shape[1])), columns=list(I.index))
    for i in range(N):
        y = np.random.choice(y, len(y), replace=False)
        null_dist.iloc[i, :] = featimp_cor(X, y)['Importance']

    plt.figure(figsize=(12,8))
    for i in range(5):
        fig = plt.subplot(2, 3, i + 1)
        fig.hist(null_dist.iloc[:, i], bins=20)
        fig.bar(I['Importance'][i], 10, color='red', width=0.001)
        fig.set_title(I.index[i])

    plt.show()

    
if __name__ == '__main__':
    
    pass
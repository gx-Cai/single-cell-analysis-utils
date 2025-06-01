import pandas as pd
import numpy as np
import os
from random import sample
from itertools import combinations
import sklearn
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test,multivariate_logrank_test
from lifelines.calibration import survival_probability_calibration
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def Cindex_model_valid(regressor, X, y, n_sample=None):
    if type(X) != np.ndarray:
        X = X.values
    if type(y) != np.ndarray:
        y = y.values

    CIndex = 0
    pairs = list(combinations(range(X.shape[0]), 2))
    if n_sample is None:
        n_sample = len(pairs)
    for i, j in sample(pairs, n_sample):
        ypredi = regressor.predict(X[i, :].reshape(1, -1))
        ypredj = regressor.predict(X[j, :].reshape(1, -1))
        if ((ypredi-ypredj)*(y[i]-y[j]) > 0) or ((ypredi == ypredj) and (y[i] == y[j])):
            CIndex += 1
    return CIndex / n_sample

def Cindex_train_valid(genenames, df_valid, df_train, model=SVR()):
    return Cindex_model_valid(
        model.fit(df_train.loc[:,genenames].values,df_train['OS.time'].values),
        X = df_valid.loc[:,genenames],
        y = df_valid['OS.time']
)

def Cindex_train(df_coxreg, gene_set, RFR = SVR(),cv=KFold(shuffle=True)):

    gene_set_ = pd.Series(gene_set)
    gene_set_ = gene_set[gene_set.isin(df_coxreg.columns)]
        
    X = df_coxreg.loc[:, gene_set_]
    y = df_coxreg['OS.time']
    return cross_val_score(
        RFR, X.values, y.values,
        scoring=Cindex_model_valid,
        cv=cv
        ).mean()

def KMFFitPlot(df_coxreg, label,colors:dict=None,ax=None,fitter_args = {},**plotargs):
    """Plotting Survival Curve by Kaplan-Meier Fitter.

    Args:
        df_coxreg (pd.DataFrame): Used for the Fitter. ROWS: samples; COLUMNS: 'OS.time','OS' and genes.
        label (pd.Series): devided samples to 2 groups.
        fitter_args (dict, optional): kwargs for KaplanMeierFitter. Defaults to {}.
        **plotargs (optional): kwargs for plot. Defaults to {}.
    """
    if ax is None:
        ax = plt.gca()
    kmf = KaplanMeierFitter(**fitter_args)
    label_unique = label.unique()
    for l in label_unique:
        ix = (label == l)
        kmf.fit(df_coxreg['OS.time'][ix], df_coxreg['OS'][ix], label=f"{l} (n={ix.sum()})")
        if colors is None:
            kmf.plot(ax=ax,**plotargs)
        else:
            kmf.plot(
                ax=ax,
                c=colors[l],
                **plotargs)
    return ax

def CPHfited(df_coxreg,**kwargs):
    cph = CoxPHFitter(**kwargs)
    cph.fit(df_coxreg, 'OS.time', event_col='OS')
    return cph.summary[cph.summary['p'] <= 0.05]

def QuantileSampleSplit(df_coxreg,gene,q=0.25,verbose=True,**plotargs):
    """From a gene to split samples to 2 groups(upper q & lower q); And copare OS with two Groups.

    Args:
        df_coxreg (DataFrame): Inputs;RAW:samples;COL:genes and OS, OS.time
        gene (str): gene should contained in df_coxreg;s Columns
        q (float, optional): ratio of Groups. Defaults to 0.25.
    """
    upper = df_coxreg.loc[:,gene].quantile(1-q)
    lower = df_coxreg.loc[:,gene].quantile(q)
    upper_ix = df_coxreg.loc[:,gene] >= upper
    lower_ix = df_coxreg.loc[:,gene] <= lower
    df_upper = df_coxreg.loc[upper_ix,[gene,'OS.time','OS']]
    df_upper['label'] = f'upper {int(q*100)}%'
    df_lower = df_coxreg.loc[lower_ix,[gene,'OS.time','OS']]
    df_lower['label'] = f'lower {int(q*100)}%'
    df = pd.concat([df_lower,df_upper])
    try:
        results = logrank_test(
            durations_A=df_upper['OS.time'],
            event_observed_A=df_upper['OS'],
            durations_B=df_lower['OS.time'],
            event_observed_B=df_lower['OS'],
        )
    except Exception as e:
        print(df_lower,df_upper)
        raise e
    p_value = results.p_value

    if verbose:
        KMFFitPlot(df,df['label'],**plotargs)
        print(f"p-value: {p_value}")
    return p_value

def QuantileSampleSplitPlot(df_coxreg,gene,ax,q=0.1,custom_labels=None,**plotargs):
    if custom_labels is None:
        labels = ['High','Low']
    
    upper = df_coxreg.loc[:,gene].quantile(1-q)
    lower = df_coxreg.loc[:,gene].quantile(q)
    upper_ix = df_coxreg.loc[:,gene] >= upper
    lower_ix = df_coxreg.loc[:,gene] <= lower
    df_upper = df_coxreg.loc[upper_ix,[gene,'OS.time','OS']]
    df_upper['label'] = labels[0]
    df_lower = df_coxreg.loc[lower_ix,[gene,'OS.time','OS']]
    df_lower['label'] = labels[1]
    df = pd.concat([df_upper,df_lower])
    try:
        results = logrank_test(
            durations_A=df_upper['OS.time'],
            event_observed_A=df_upper['OS'],
            durations_B=df_lower['OS.time'],
            event_observed_B=df_lower['OS'],
        )
    except Exception as e:
        print(df_lower,df_upper)
        raise e
    p_value = results.p_value
    
    colors = plotargs.get(
        'colors',
        {
            labels[0]:'#ff7f0e',
            labels[-1]:'#1f77b4',
        }
    )
    if 'colors' in plotargs.keys():
        del plotargs['colors']
    KMFFitPlot(df,df['label'],ax=ax, colors = colors, **plotargs)
    ax.plot([],[],label=f"p-value = {p_value:.3f}",color='white')
    ax.legend(
        frameon=False,loc='best'
    )
    ax.set_title(gene)
    ax.set_xlabel('')
    return ax

def QuantileSampleTripleSplitPlot(df_coxreg,gene,ax,q=0.1,custom_labels=None,**plotargs):
    if custom_labels is None:
        labels = ['High','Moderate','Low']
        
    upper = df_coxreg.loc[:,gene].quantile(1-q)
    lower = df_coxreg.loc[:,gene].quantile(q)
    upper_ix = df_coxreg.loc[:,gene] >= upper
    lower_ix = df_coxreg.loc[:,gene] <= lower
    moderate_ix = (df_coxreg.loc[:,gene] < upper) & (df_coxreg.loc[:,gene] > lower)
    df = df_coxreg.copy()
    df.loc[upper_ix,'label'] = labels[0]
    df.loc[lower_ix,'label'] = labels[-1]
    df.loc[moderate_ix,'label'] = labels[1]
    results = multivariate_logrank_test(
        event_durations=df['OS.time'],
        event_observed=df['OS'],
        groups=df['label'],
    )
    p_value = results.p_value
    
    colors = plotargs.get(
        'colors',
        {
        labels[0]:'#ff7f0e',
        labels[-1]:'#1f77b4',
        labels[1]:'.5',
        }
    )

    KMFFitPlot(df,df['label'],ax=ax,colors=colors,**plotargs)
    ax.plot([],[],label=f"p-value = {p_value:.3f}",color='white')
    ax.legend(
        frameon=False,loc='best'
    )
    ax.set_title(gene)
    ax.set_xlabel('')
    return ax

def GreedyGeneSelect(df_coxreg, sorted_gene_list,scorer = Cindex_train):
    select_genes=[]
    best_score = -1
    while True:
        flag = 1
        temp_best_score = best_score
        for gene in sorted_gene_list:
            if gene not in df_coxreg.columns:continue
            if gene in select_genes:continue
            tmpt_selected = select_genes + [gene]
            score = scorer(df_coxreg, tmpt_selected)
            if score > temp_best_score:
                temp_best_score = score
                # select_genes.append(gene)
                best_gene = gene
                flag = 0
        if flag: break
        best_score = temp_best_score
        select_genes.append(best_gene)
        print(len(select_genes),best_score,)
    return select_genes

def ScatterCoxPlot(
    df_inputs,
    annote_x_adjust=0,
    annote_y_adjust=0,
    max_x=None,
    ):
    model = CoxPHFitter().fit(df_inputs.loc[:,['OS','OS.time','OSscore']], 'OS.time', event_col='OS')
    coef = model.summary.loc['OSscore','coef']
    p = model.summary.loc['OSscore','p']
    cidx = model.concordance_index_
    time_sort_sample = df_inputs.loc[:,'OS.time'].sort_values().index
    time_idx = df_inputs.loc[time_sort_sample,'OS.time'].values
    harzard_ratio = np.cumsum(df_inputs.loc[time_sort_sample,'OS'].values) / df_inputs.shape[0]

    f = plt.figure(dpi=300)
    ax = f.add_subplot(111)
    ax.scatter(
        df_inputs.loc[:,"OS.time"],
        # np.exp(coef*(df_inputs.loc[:,"OSscore"]-df_inputs.loc[:,"OSscore"].mean())),
        df_inputs.loc[:,"OSscore"],
        c = df_inputs.loc[:,"OS"],
        cmap = 'coolwarm',
    )
    # ax.set_yscale('log')
    # ax.set_yticks([])
    ax.set_ylabel(f'OS score')
    ax.set_xlabel('Time(days)')
    if max_x:
        ax.set_xlim(0,max_x)

    ax2 = ax.twinx()
    ax2.plot(
        time_idx,
        harzard_ratio,
        color='k',
        linewidth=1,linestyle='--')
    ax2.set_ylabel('Cumulative Harzard Ratio')

    # annote
    cm = plt.get_cmap('coolwarm')
    ax.text(
        annote_x_adjust+0.05,annote_y_adjust+0.93,
        'Death',
        transform = ax.transAxes,
        color = cm(0.999),
        fontsize = 12,
        fontweight = 'bold'
    )
    ax.text(
        annote_x_adjust+0.17,annote_y_adjust+0.93,
        'Alive',
        transform = ax.transAxes,
        color = cm(0),
        fontsize = 12,
        fontweight = 'bold'
    )

    ax.text(
        annote_x_adjust+0.05,annote_y_adjust+0.73,
        f"UNIVARIATE COX MODEL\np-value = {p:.3f}\nc-index = {cidx:.3f}\ncoef = {coef:.3f}",
        transform = ax.transAxes,
    )
    ax.set_ylabel('PM-risk score')

    return f

def PartialEffectPlot(df_inputs, var):

    model = CoxPHFitter().fit(df_inputs.loc[:,['OS','OS.time',var]], duration_col='OS.time', event_col='OS')
    f,ax = plt.subplots(1,1,dpi=300)
    model.plot_partial_effects_on_outcome(var, values=np.linspace(-0.5, 0.5, 5), cmap='coolwarm',ax=ax)
    ax.legend(
        edgecolor='w',
        )
    ax.set_ylabel('Survival probability')
    ax.set_xlabel('Time(days)')
    return f

def CarlibrateCoxPlot(df_inputs,t0=None):
    training_data = df_inputs.loc[:,['OS','OS.time','OSscore']]
    model = CoxPHFitter().fit(training_data, duration_col='OS.time', event_col='OS')
    f,ax = plt.subplots(1,1,dpi=300)
    ax,ici,e50 = survival_probability_calibration(
        model,
        training_data,
        t0=t0 if t0 is not None else training_data['OS.time'].mean(),
        ax=ax
        )
    ax.legend(
        edgecolor='w',
        loc='best')
    return f


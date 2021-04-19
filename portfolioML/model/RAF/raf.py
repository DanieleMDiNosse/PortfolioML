'''Random Forest'''
import argparse
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from portfolioML.data.data_returns import read_filepath
from portfolioML.model.split import all_data_DNN
from makedir import smart_makedir, go_up
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def predictions_roc(n_estimators, max_depth, num_period, criterion):
    '''
    Parameters
    ----------
    n_estimators : Int
        DESCRIPTION. Number of threes in the random forest
    num_period : TYPE, optional
        DESCRIPTION. The default is 10.
    -------
    Plot roc curve with mean and standard deviation of the area under the curve (auc) of a
    trained model.
    Each model is trained over several study period so its name contain this information,
    for semplicity put this information in this way: '..._periond#' (for istance '_period0').
    Nember of period running over several argumenti setting in 'periods' argument.
    So before running thi function carefully checking of the folder is suggest to avoid
    problems.

    Tecnical aspects: because of each model has got different tpr and fps, an interpoletion
    of this values is used in order to have the same leght for each model.

    Parameters
    ----------
    model: string
    File path of the model, it must have in the name 'periond' (for istance 'period0') to
    take in mind the numer of perion over wich the model is trained

    periods: integer
    Numer of model that are taken in order to compute plot and final values
    '''

    path_n = go_up(level_up=2)
    path_p = path_n + f'/results/predictions/RAF/{args.name_model}/'
    path_r = path_n + f'/results/ROC/RAF/{args.name_model}/'

    tpr_list = []
    aucs_list = []
    interp_fpr = np.linspace(0, 1, 10000)
    plt.figure()

    for i in range(0, num_period):
        x_train, y_train, x_test, y_test = all_data_DNN(df_returns, df_binary, i)
        rf_ = RandomForestClassifier(n_estimators, criterion, max_depth, n_jobs=-1,
                                    min_samples_split=2, min_samples_leaf=1,
                                    random_state=10, verbose=1)
        rf_.fit(x_train, y_train)
        y_proba = rf_.predict_proba(x_test)


        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1])
        interp_tpr = np.interp(interp_fpr, fpr, tpr)
        tpr_list.append(interp_tpr)
        roc_auc = roc_auc_score(y_test, y_proba[:,1])
        aucs_list.append(roc_auc)

        plt.plot(fpr, tpr, label=f'per{i} (area = %0.4f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate',)
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", fontsize=12, frameon=False)
        plt.title(f'ROC CURVE')
        plt.savefig(path_r + f'ROC_CURVE.png')

        y_pred_companies = [y_proba[i:87+i] for i in range(0,len(y_proba)-87+1,87)]
        dict_comp = {df_returns.columns[i]: y_pred_companies[i] for i in range(0,365)}
        df_predictions = pd.DataFrame()
        for tick in df_returns.columns:
            df_predictions[tick] = dict_comp[tick][:,0]
            df_predictions.to_csv(path_p + f'RAF_{args.name_model}_Predictions_{i}th_Period.csv')

    auc_mean = np.mean(np.array(aucs_list))
    auc_std = np.std(np.array(aucs_list))
    tpr_mean = np.mean(tpr_list, axis=0)

    plt.figure()
    plt.plot(interp_fpr, tpr_mean, color='b',
            label=fr'Mean ROC (AUC = {auc_mean:.4f} $\pm$ {auc_std:.4f})', lw=1, alpha=.8)
    tpr_std = np.std(tpr_list, axis=0)
    tprs_upper = np.minimum(tpr_mean + tpr_std, 1)
    tprs_lower = np.maximum(tpr_mean - tpr_std, 0)
    plt.fill_between(interp_fpr, tprs_lower, tprs_upper, color='blue', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize=12, frameon=False)
    plt.title(f'ROC Curve - mean +|- std')
    plt.savefig(path_r + f'ROC Curve - mean +|- std')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RandomForestClassifier')
    parser.add_argument("-log", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info"))
    parser.add_argument('-num_period', type=int, default=10, help='Number of periods (default=10)')
    parser.add_argument('-n_estimators', type=int, default=1000, help='Number of trees (default=1000)')
    parser.add_argument('-criterion', type=str, default='gini', help='Criterion (default=gini)')
    parser.add_argument('-max_depth', type=int, default=25, help='Trees\'s depth (default=25) ')
    parser.add_argument('name_model', type=str, help='Name_model')
    args = parser.parse_args()

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}

    logging.basicConfig(level= levels[args.log])
    pd.options.mode.chained_assignment = None

    df_returns_path = go_up(2) + "/data/ReturnsData.csv"
    df_binary_path = go_up(2) + "/data/ReturnsBinary.csv"
    df_returns = read_filepath(df_returns_path)
    df_binary = read_filepath(df_binary_path)

    smart_makedir(f'/results/predictions/RAF/{args.name_model}/', level_up=2)
    smart_makedir(f'/results/ROC/RAF/{args.name_model}/', level_up=2)

    predictions_roc(n_estimators=args.n_estimators, max_depth=args.max_depth, num_period=args.num_period, criterion=args.criterion)
    with open(f"{args.name_model}.txt", 'a', encoding='utf-8') as file:
        file.write(f'num_period={args.num_period}, n_estimators={args.n_estimators}, criterion={args.criterion},max_depth={args.max_depth}' + '\n')
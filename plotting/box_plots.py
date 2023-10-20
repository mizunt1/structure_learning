import sys, os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

def plot(base_dir, method_path, method_str, ordered_str,fig_path):
    method_dict = dict(zip(method_str, method_path))
    result = "results2"
    df = pd.read_csv(base_dir + '/' + method_path[0], index_col=0)
    #metrics  = ['nll']
    metrics = df.columns.to_list()
    dfs = {}
    # for all metrics, put in dataframe with name of method
    for metric in metrics:
        data_for_metric = {}
        for method in ordered_str:
            stri = method_dict[method]
            file_path = base_dir + '/' + stri
            data = pd.read_csv(file_path, index_col=0)[metric]
            data_for_metric[method] = data.to_list()
        dfs[metric] = pd.DataFrame.from_dict(data_for_metric, orient="index")

    
    # sns.set(font="Verdana")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    font = {'family' : 'serif',
            'size'   : 12}

    mpl.rc('font', **font)
    
    for metric in metrics:

        plt.figure(figsize=(6,5.7), dpi=100)
        plt.clf()
        print(metric)
        if metric == 'nll':
            plot_is = sns.boxplot(data=(dfs[metric].drop('MH').T), showfliers = False)
        else:
            plot_is = sns.boxplot(data=(dfs[metric].T))
        plt.xticks(rotation=20)
        data = dfs[metric].T

        quantiles = data.quantile([0.5, 0.25, 0.75])
        metric = metric.replace(" ", "_")
        if metric == 'nll':
            import pdb
            pdb.set_trace()
        quantiles.to_csv(fig_path + '/' + metric + ".csv")
        fig = plot_is.get_figure()
        fig.savefig(fig_path +'/' + metric + ".png")

if __name__ == '__main__':
    nodes_5 = False
    if nodes_5:
        base_dir = 'aistats_5'
        methods = ['ges_arxiv2_n5', 'bcd_arxiv2', 'dibs_plus_arxiv2_n5', 'pc_arxiv2_n5', 'dibs_arxiv2_n5', 'gibs_arxiv2_n5',  'mh_arxiv2_n5','vbg_arxiv2_w_0.5', 'jsp_5_correct2', 'dag_gfn_5']
    
        method_path = [method + '.csv' for method in methods]
        fig_path = 'figs_aistats_n5'
        fig_path = 'aistats_5_appendix'
    else:
        base_dir = 'aistats_20'
        method_path = ['ges_arxiv2_n20.csv', 'bcd_arxiv_n20.csv',
                       'dibs_plus_arxiv2_n20.csv', 'pc_arxiv2_n20.csv',
                       'dibs_arxiv2_n20.csv', 'gibbs_arxiv2_n20.csv', 'mh_arxiv2_n20.csv',
                       'vbg_arxiv2_w_0.5_n20.csv', 'jsp_target.csv', 'dag_gfn_20.csv']
        fig_path = 'figs_aistats_n20/'
        fig_path = 'aistats_20_appendix'

    method_str = ['BS GES', 'bcd nets', 'DiBS +', 'BS PC', 'DiBS', 'Gibbs', 'MH', 'VBG', 'JSP', 'DAG-GFN']
    ordered_str = ['VBG','DAG-GFN', 'JSP','DiBS', 'DiBS +', 'bcd nets',
                   'BS GES',  'BS PC',  'Gibbs', 'MH']
    #, 'MH_burn', 'MH_theta']

    plot(base_dir, method_path, method_str, ordered_str, fig_path)

import wandb
import json
import sys, os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd


api = wandb.Api(timeout=19)
entity, project = 'mizunt', 'sachs_20'
def get_runs():
    return api.runs(entity + '/' + project,
        filters={'$and': [{
            'state': {'$eq': 'finished'},
        }]}
    )

#methods = ['vbg', 'dag_gflownet', 'dibs', 'dibs +', 'dibs marg', 'bcd','ges', 'pc', 'gibbs', 'mh']
#method_str = ['VBG', 'DGFN', 'DiBS', 'DiBS+', 'DiBS M', 'BCD Nets',
#                   'BS GES',  'BS PC',  'Gibbs', 'MH']

methods = ['vbg', 'dag_gflownet', 'jsp_target','dibs', 'dibs +', 'bcd','ges', 'pc', 'gibbs', 'mh']
method_str = ['VBG','DAG-GFN','JSP', 'DiBS', 'DiBS +','BCD Nets',
                   'BS GES',  'BS PC',  'Gibbs', 'MH']

method_zip = dict((zip(methods, method_str)))
metrics = ['auroc', 'e-shd']
runs = get_runs()
results = {}
num_seeds = 5
for metric in metrics:
    results[metric] = {}
    for method in method_str:
        seeds = [str(i) for i in range(num_seeds)]
        results[metric][method] = dict(zip((seeds), [None for i in seeds]))

for run_ in runs:
    inf = json.loads(run_.json_config)
    method = inf['model']['value']
    if method == 'dibs':
        if inf['plus']['value']:
            method = 'dibs +'
        else:
            method = 'dibs'
            if 'marginal' in inf.keys():
                if inf['marginal']['value']:
                    method = 'dibs marg'
        inf['model']
    if method == 'mcmc':
        if inf['method']['value'] == 'mh':
            method = 'mh'
        else:
            method = 'gibbs'
    if method == 'bs':
        if inf['method']['value'] == 'pc':
            method = 'pc'
        else:
            method = 'ges'
    if method == 'jsp':
        if inf['update_target_every']['value'] == 100:
            method = 'jsp_target'
        else:
            method = 'jsp'
            continue
    method = method_zip[method]
    try: 
        auroc = run_.summary['metrics/thresholds']['roc_auc']
    except:
        continue
    shd = run_.summary['metrics/shd/mean']
    seed = inf['seed']['value']
    
    results['auroc'][method][str(seed)] = auroc
    results['e-shd'][method][str(seed)] = shd
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
font = {'family' : 'serif',
        'size'   : 12}

mpl.rc('font', **font)
fig_path = 'sachs_appendix'
for metric in metrics:
    plt.figure(figsize=(6,5.7), dpi=100)
    plt.clf()
    print(metric)
    df = pd.DataFrame(results[metric])
    plot_is = sns.boxplot(data=(df))
    plt.xticks(rotation=20)
    data = df
    quantiles = data.quantile([0.5, 0.25, 0.75])
    metric = metric.replace(" ", "_")
    quantiles.to_csv(fig_path + '/' + metric + ".csv")
    fig = plot_is.get_figure()
    fig.savefig(fig_path +'/' + metric + ".png")
    
    

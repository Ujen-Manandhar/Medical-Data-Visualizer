import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv', index_col=0)

def bmi(row):
    if (row.weight / (row.height*0.01)**2) > 25:
        return 1
    return 0

# 2
df['overweight'] = df.apply(bmi, axis=1)

# 3 Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = df.cholesterol.map(lambda x : 0 if x == 1 else 1)
df['gluc'] = df.gluc.map(lambda x : 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active','overweight'])


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat =df_cat.rename(columns={0:'total'})

    # 7
    df_cat.head()

    
    # 8
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue= 'value', col='cardio', kind='bar').fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.copy()

    df_heat = df_heat.loc[
    (df_heat['ap_lo'] <= df_heat['ap_hi']) &
    (df_heat['height'] >= df_heat['height'].quantile(0.025))&
    (df_heat['height'] <= df_heat['height'].quantile(0.975))&
    (df_heat['weight'] >= df_heat['weight'].quantile(0.025))&
    (df_heat['weight'] <= df_heat['weight'].quantile(0.975))]

    # 12
    corr = df_heat.reset_index().corr()

    # 13
    mask = np.triu(corr) # include the diagonal 
    corr = corr[corr != mask]


    # 14
    fig, ax = plt.subplots(figsize = (12, 12))

    # 15
    sns.heatmap(corr, annot=True, fmt='.1f', linewidths=1)


    # 16
    fig.savefig('heatmap.png')
    return fig

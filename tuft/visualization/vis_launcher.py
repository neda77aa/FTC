# launcher for the dataframe analysis and visualization
import pandas as pd
import numpy as np
from vis import plot_confusion_matrix, aggregate_dataframe
from vis import plot_rejection_characteristic, find_rejection_characteristic

mildmod_scheme = [0,1,1,2]
all_scheme = [0,1,2,3]

def plot_AS_confusion(df, run_name, task_name, classes):
    true = np.array(df['GT_AS'])
    pred = np.array(df['pred_AS'])
    plot_confusion_matrix(true, pred, run_name, task_name, classes)
    
# #%% plotting rejection characteristic for 2 runs
# df1 = pd.read_csv(r"D:\Projects\AS\logs\sparkling-snowflake-24\test.csv")
# df2 = pd.read_csv(r"D:\Projects\AS\logs\pious-fog-34\test.csv")
# df1 = df1[df1['GT_B']==1]
# df2 = df2[df2['GT_B']==1]
# true1 = np.array(df1['GT_AS'])
# pred1 = np.array(df1['pred_AS'])
# count1, acc1 = find_rejection_characteristic(true1, pred1, np.array(df1['ent_AS']))
# count4, acc4 = find_rejection_characteristic(true1, pred1, 1-np.array(df1['max_AS']))
# true2 = np.array(df2['GT_AS'])
# pred2 = np.array(df2['pred_AS'])
# count2, acc2 = find_rejection_characteristic(true2, pred2, np.array(df2['ent_AS']))
# count3, acc3 = find_rejection_characteristic(true2, pred2, np.array(df2['vac_AS']))
# labels = ['softmax entropy', 'evidential entropy', 'evidential vacuity', 'softmax confidence']
# plot_rejection_characteristic([count1, count2, count3, count4], [acc1, acc2, acc3, acc4], 
#                               labels, "softmax vs evidential, bicuspid")
#%% plotting confusion performance for 1 run
run_name = "sparkling-snowflake-24"
csv_filepath = r"D:\Projects\AS\logs\sparkling-snowflake-24\test.csv"
view = "all"
df = pd.read_csv(csv_filepath)

# use mildmod scheme if indicated
division = "all"
if division == "mildmod":
    scheme = [0,1,1,2]
    classes = ['normal','mild/mod','severe']
else:
    scheme = [0,1,2,3]
    classes = ['normal','mild','moderate','severe']
df['GT_AS'] = df['GT_AS'].map(lambda x: scheme[x])
df['pred_AS'] = df['pred_AS'].map(lambda x: scheme[x])

dfp = aggregate_dataframe(df, strategy="majority", include_quality=False)
classes_B = ['tricuspid', 'bicuspid']

#all cohorts
true_B = np.array(df['GT_B'])
pred_B = np.array(df['pred_B'])
plot_confusion_matrix(true_B, pred_B, run_name, view+" Bicuspid", classes_B)

plot_AS_confusion(df, run_name, view+" AS", classes)

# AS of bicuspid cases
df_B = df[df['GT_B']==1]
plot_AS_confusion(df_B, run_name, view+" AS (Bicuspid only)", classes)

# all cohorts, patient-level
plot_AS_confusion(dfp, run_name, view+" AS (Patient-level)", classes)
dfp_B = dfp[dfp['GT_B']==1]
plot_AS_confusion(dfp_B, run_name, view+" AS (Patient-level, Bicuspid only)", classes)

# # AS of tricuspid cases
# df_T = df[df['GT_B']==0]
# plot_AS_confusion(df_T, run_name, view+" AS (Tricuspid only)")

# # 60+ only
# df_60_plus = df[df['age']>=60]
# plot_AS_confusion(df_60_plus, run_name, view+" AS (60+ only)")

# # <60 only
# df_young = df[df['age']<60]
# plot_AS_confusion(df_young, run_name, view+" AS (<60 only)")
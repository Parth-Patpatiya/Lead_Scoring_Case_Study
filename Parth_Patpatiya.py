#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 60)


# In[7]:


from sklearn.feature_selection import RFE


# In[8]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate


# In[9]:


from sklearn.linear_model import LogisticRegression


# In[10]:


from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score, accuracy_score


# In[21]:


df = pd.read_csv("Leads.csv")
df.head()


# In[16]:


df.info()


# In[17]:


df.describe()


# In[24]:


df.columns = ['_'.join(name.lower().split()[:3]) for name in df.columns]
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.str.lower().str.replace(' ', '_').str.strip())
for col in df.columns[2:]:
    print(col.upper(), end=': ')
    print(df[col].unique())
    print()


# In[26]:


df


# In[88]:


df = df.replace('select', np.nan)
df


# In[91]:


df = df.replace('select', np.nan)
df


# In[93]:


df.tags = df.tags.replace("wrong_number_given", "invalid_number")
df


# In[35]:


value_counts = df.lead_source.value_counts()
value_counts


# In[36]:


df.lead_source = df.lead_source.replace(value_counts[value_counts < 35].index, "others")


# In[38]:


value_counts_1 = df.country.value_counts()
value_counts_1 


# In[94]:


df.country = df.country.replace(df.country[df.country != 'india'].dropna().unique(), "Different")
df


# In[97]:


round(df.isna().sum().sort_values(ascending=False)/len(df)*100, 3)
df


# In[ ]:


df.drop('how_did_you', axis=1, inplace=True)


# In[101]:


df.lead_quality.fillna("not_sure", inplace=True)
print(df.lead_quality.value_counts())


# In[98]:


dummies = pd.get_dummies(df.select_dtypes(include=['object']), drop_first=True)


# In[49]:


New_df1 = df.drop(df.select_dtypes(include=['object']).columns, axis=1)
New_df1 = pd.concat([clean_df, dummies], axis=1)

New_df1.head()


# In[50]:


plt.figure(figsize=(20, 15))
sns.heatmap(New_df1.corr())
plt.show()


# In[51]:


sns.heatmap(New_df1[New_df1.columns[:4]].corr())
plt.show()


# In[53]:


def plot_bars():
    plt.figure(figsize=(20, 15))
    plt.subplot(121)
    sns.distplot(New_df1['total_time_spent'])

    plt.subplot(122)
    sns.distplot(New_df1['totalvisits'])

    plt.tight_layout()
    plt.show()
    
plot_bars()


# In[55]:


total_columns = New_df1[['converted', 'totalvisits',  'page_views_per', 'total_time_spent']]

def plot_boxes():
    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    sns.boxplot(data=New_df1, x='total_time_spent')

    plt.subplot(122)
    sns.boxplot(data=New_df1, x='totalvisits')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    box_long = pd.melt(total_columns.drop('total_time_spent', axis=1), id_vars='converted')
    sns.boxplot(x='converted', y='value', hue='variable', data=box_long)
    plt.show()
    
plot_boxes()


# In[66]:


X = New_df1.drop('converted', axis=1)
y = New_df1['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.40, random_state=50)


# In[67]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler


# In[69]:


scaler = MinMaxScaler()
columns_new = X_train.columns
scaling_X_train = pd.DataFrame(scaler.fit_transform(X_train[columns_new[:3]]), columns=columns_new[:3])
scaling_X_train = pd.concat([scaled_X_train, X_train.drop(cols[:3], axis=1).reset_index(drop=True)], axis=1)
scaling_X_test = pd.DataFrame(scaler.transform(X_test[cols[:3]]), columns=cols[:3])
scaling_X_test = pd.concat([scaling_X_test, X_test.drop(cols[:3], axis=1).reset_index(drop=True)], axis=1)


# In[72]:


scaling_X_train,scaling_X_test


# In[81]:


def optimize_varaibles(a, b):
    
    optimized = list()
    for features in range(a, b):
        log_reg = LogisticRegression(C=2, random_state=42)
        rfe = RFE(log_reg, features)
        rfe.fit(scaled_X_train, y_train)
        cols = scaled_X_train.columns[rfe.support_]
        


# In[87]:


def get_vif():
    vif = pd.DataFrame()
    vif['Features'] = scaling_X_train[cols].columns
    vif['VIF'] = [variance_inflation_factor(scaling_X_train[cols].values, i) for i in range(scaling_X_train[cols].shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif
    
get_vif()


# In[ ]:


X_train_sm = sm.add_constant(scaling_X_train[cols])
logm2 = sm.GLM(list(y_train), X_train_sm, family=sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


def draw_roc(actual_values, probability_estimates):
    fpr, tpr, thresholds = roc_curve(actual_values, probability_estimates, drop_intermediate=False)
    auc_score = roc_auc_score(actual_values, probability_estimates)
    
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {round(auc_score, 2)})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def draw_prec_recall(actual_values, probability_estimates):
    
    p, r, thresholds = precision_recall_curve(actual_values, probability_estimates)
    plt.plot(thresholds, p[:-1], "b-", label="Precision")
    plt.plot(thresholds, r[:-1], "r-", label="Recall")
    plt.title("Precison - Recall Trade off")
    plt.legend(loc="lower right")
    plt.show()

def get_metrics(y, pred, prob_est):
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f = f1_score(y, pred)

    # Sensitivity, Specificity
    print(f"Sensitivity (Recall): {recall}\nSpecificity: {tn/(tn+fp)}\nPrecision: {precision}\nF-Score: {f}")

    # Reciever Operating Characteristic Curve
    draw_roc(y, prob_est[:, 1])

    # Precision Recall Curve
    draw_prec_recall(y, prob_est[:, 1])
    
get_metrics(y_train, pred, prob_est)


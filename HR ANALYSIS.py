#!/usr/bin/env python
# coding: utf-8
pwdpwd
# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv(r'C:\Users\Lenovo\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


col=['Attrition','BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime',
     'DistanceFromHome','EmployeeNumber','EnvironmentSatisfaction','JobInvolvement','JobLevel','NumCompaniesWorked','PerformanceRating',
     'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany',
     'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[11]:


plt.figure(figsize=(20,55))
for i in range(0,len(col)):
    plt.subplot(13,3,i+1)
    sns.countplot(df[col[i]])


# In[12]:


plt.figure(figsize=(12,8))
sns.barplot(x='Age',y='MonthlyIncome',data=df)


# In[13]:


plt.figure(figsize=(18,8))
sns.barplot(x='Department',y='MonthlyIncome',data=df,hue='EducationField')


# In[14]:


plt.figure(figsize=(12,8))
sns.barplot(x='Age',y='MonthlyRate',data=df)


# In[15]:


from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()


# In[16]:


for i in  df.columns:
    if df[i].dtypes=="object":
        df[i]=enc.fit_transform(df[i].values.reshape(-1,1))


# In[17]:


df


# In[18]:


df.describe()


# In[19]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.title('DESCRIBE')
sns.heatmap(df.describe(), annot=True, linewidth=0.56, linecolor='green', fmt ='.1f')


# In[20]:


df.corr()


# In[21]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.title('CORRELATION')
sns.heatmap(df.corr(), annot=True, linewidth=0.56, linecolor='green', fmt ='.1f')


# In[22]:


plt.figure(figsize=(15,20))
for i in enumerate(df):
    plt.subplot(13,3,i[0]+1)
    sns.boxplot(i[1],data=df)
    plt.tight_layout()


# In[23]:


for col in df.columns:
    percentile=df[col].quantile([0.01,0.98]).values
    df[col][df[col]<=percentile[0]]=percentile[0]
    df[col][df[col]>=percentile[1]]=percentile[1]


# In[24]:


plt.figure(figsize=(15,20))
for i in enumerate(df):
    plt.subplot(13,3,i[0]+1)
    sns.boxplot(i[1],data=df)
    plt.tight_layout()


# In[25]:


x = df.drop('Attrition',axis=1)


# In[26]:


y = df["Attrition"]


# In[38]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)
x


# In[39]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.metrics import accuracy_score


# In[30]:


for i in range(0,1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=i,test_size=0.20)
    lr.fit(x_train,y_train)
    pred_train=lr.predict(x_train)
    pred_test=lr.predict(x_test)
    if round(accuracy_score(y_train,pred_train)*100,1)==round(accuracy_score(y_test,pred_test)*100,1):
        print("At random state",i,"The model performs well")
        print("At random_state:-",i)
        print ("Training accuracy is :-",accuracy_score(y_train,pred_train)*100)
        print("Testing r2_score is:-", accuracy_score(y_test,pred_test)*100)


# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=35,test_size=0.20,)


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_test))


# In[33]:


pred_lr = lr.predict(x_test)
from sklearn.model_selection import cross_val_score
lss = accuracy_score(y_test,pred_lr)
for j in range(2,11):
    lsscore = cross_val_score(lr,x,y,cv=j)
    lsc = lsscore.mean(0)
    print ("At cv:-",j)
    print("Cross validation score is:-",lsc*100)
    print("accuracy_score is:-",lss*100)
    print("\n")
    


# In[42]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(pred_test, y_test)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr, tpr, color= 'darkorange', lw=10, label='ROC curve (area=%0.2f)' % roc_auc)
plt.plot([0, 1],[0, 1], color='navy', lw=10, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('false Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc= 'lower right')
plt.show()


# In[35]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[36]:


from sklearn.linear_model import Lasso

parameters = {'alpha':[.0001, .001, .01, 1, 10],'random_state':list(range(0,10))}
ls = Lasso()
clf = GridSearchCV(ls,parameters)
clf.fit(x_train,y_train)

print(clf.best_params_)


# In[51]:


from sklearn.metrics import r2_score
ls = Lasso(alpha=1,random_state=0)
ls.fit(x_train,y_train)
ls.score(x_test,y_test)
pred_ls = ls.predict(x_test)

lss = r2_score(y_test,pred_ls)
lss


# In[ ]:





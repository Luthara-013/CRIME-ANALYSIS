# CRIME-ANALYSIS
#correlation analysis of crimes against women in different states of india

# In[57]:
import scipy.stats as st
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.chdir('G:/SET PROJECT/crime rate prediction/dataset')

# In[127]:

data=pd.read_csv('factors of women.csv')
data


# In[128]:


data.info()


# In[129]:


data.describe()


# # CORRELATION

# In[130]:


data.corr().round(2)


# In[131]:


data['RAPE'].corr(data['KIDNAP & ABDUCTION'])


# In[48]:


data['DOWRY DEATHS'].corr(data['CRUETLY BY HUSBAND OR HIS RELATIVES'])


# In[49]:


data['HUMAN TRAFFICKING'].corr(data['IMORTAL TRAFFICKING'])


# In[50]:


data['SEXUAL HARRASSMENT'].corr(data['ASSAULT ON MODESTY'])


# # correlation using numpy
# 

# In[51]:


np.corrcoef(data['RAPE'],data['KIDNAP & ABDUCTION'])
#correlation between rape and kidnap&abduction is 0.71669125
#it has a moderate positive correlation


# In[52]:


np.corrcoef(data['DOWRY DEATHS'],data['CRUETLY BY HUSBAND OR HIS RELATIVES'])
#correlation between dowry deaths and cruelty by husband or his relatives is 0.48595863
#it has a moderate positive correlation


# In[53]:


np.corrcoef(data['HUMAN TRAFFICKING'],data['IMORTAL TRAFFICKING'])
#correlation between human trafficking and immortal trafficking is 0.05271316
#it has moderate positive correlation


# In[54]:


np.corrcoef(data['SEXUAL HARRASSMENT'],data['ASSAULT ON MODESTY'])
#correlation between sexual harrasssment and  assualt on modesty is 0.53994416
#it has a moderate positive correlation


# # correlation using scipy

# #### RAPE AND KIDNAP&ABDUCTION

# In[55]:


st.pearsonr(data['RAPE'],data['KIDNAP & ABDUCTION'])


# In[80]:


sns.scatterplot(x='RAPE',y='KIDNAP & ABDUCTION',data=data)


# In[79]:


sns.barplot(x='RAPE',y='KIDNAP & ABDUCTION',data=data)


# In[78]:


sns.boxplot(x='RAPE',y='KIDNAP & ABDUCTION',data=data)


# In[91]:


sns.lineplot(x='RAPE',y='KIDNAP & ABDUCTION',data=data)


# In[116]:


sns.regplot(x='RAPE',y='KIDNAP & ABDUCTION',data=data)


# #### DOWRY DEATHS AND CRUELTY BY HUSBAND OR HIS RELATIVES

# In[64]:


st.pearsonr(data['DOWRY DEATHS'],data['CRUETLY BY HUSBAND OR HIS RELATIVES'])


# In[86]:


sns.scatterplot(x='DOWRY DEATHS',y='CRUETLY BY HUSBAND OR HIS RELATIVES',data=data)


# In[87]:


sns.barplot(x='DOWRY DEATHS',y='CRUETLY BY HUSBAND OR HIS RELATIVES',data=data)


# In[88]:


sns.boxplot(x='DOWRY DEATHS',y='CRUETLY BY HUSBAND OR HIS RELATIVES',data=data)


# In[89]:


sns.lineplot(x='DOWRY DEATHS',y='CRUETLY BY HUSBAND OR HIS RELATIVES',data=data)


# In[115]:


sns.regplot(x='DOWRY DEATHS',y='CRUETLY BY HUSBAND OR HIS RELATIVES',data=data)


# #### HUMAN TRAFFICKING AND IMORTAL TRAFFICKING 

# In[66]:


st.pearsonr(data['HUMAN TRAFFICKING'],data['IMORTAL TRAFFICKING'])


# In[92]:


sns.scatterplot(x='HUMAN TRAFFICKING',y='IMORTAL TRAFFICKING',data=data)


# In[93]:


sns.barplot(x='HUMAN TRAFFICKING',y='IMORTAL TRAFFICKING',data=data)


# In[94]:


sns.boxplot(x='HUMAN TRAFFICKING',y='IMORTAL TRAFFICKING',data=data)


# In[95]:


sns.lineplot(x='HUMAN TRAFFICKING',y='IMORTAL TRAFFICKING',data=data)


# In[114]:


sns.regplot(x='HUMAN TRAFFICKING',y='IMORTAL TRAFFICKING',data=data)


# ####  SEXUAL HARRASSMENT AND ASSAULT ON MODESTY

# In[99]:


st.pearsonr(data['SEXUAL HARRASSMENT'],data['ASSAULT ON MODESTY'])


# In[100]:


sns.scatterplot(x='SEXUAL HARRASSMENT',y='ASSAULT ON MODESTY',data=data)


# In[101]:


sns.barplot(x='SEXUAL HARRASSMENT',y='ASSAULT ON MODESTY',data=data)


# In[102]:


sns.boxplot(x='SEXUAL HARRASSMENT',y='ASSAULT ON MODESTY',data=data)


# In[103]:


sns.lineplot(x='SEXUAL HARRASSMENT',y='ASSAULT ON MODESTY',data=data)


# In[113]:


sns.regplot(x='SEXUAL HARRASSMENT',y='ASSAULT ON MODESTY',data=data)


# # plots for all factors

# In[96]:


sns.pairplot(data)


# # correlation heatmap 

# In[112]:


plt.figure(figsize=(10,8))
plot=sns.heatmap(data.corr().round(2),annot=True)
plt.show()


# # spearman r

# In[106]:


from scipy.stats import spearmanr


# In[107]:


spearmanr(data['RAPE'],data['KIDNAP & ABDUCTION'])


# In[108]:


spearmanr(data['DOWRY DEATHS'],data['CRUETLY BY HUSBAND OR HIS RELATIVES'])


# In[109]:


spearmanr(data['HUMAN TRAFFICKING'],data['IMORTAL TRAFFICKING'])


# In[117]:


spearmanr(data['SEXUAL HARRASSMENT'],data['ASSAULT ON MODESTY'])


#simple linear regression analysis on crimes against women in different states of india

# # SIMPLE LINEAR REGRESSION

# In[49]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy import stats
import os
os.chdir('G:/SET PROJECT/crime rate prediction/dataset')


# In[50]:


pd=pd.read_csv('factors of women.csv')


# #### regression line of kidnap&abduction on rape(y=a+bx)

# In[51]:


features=pd['RAPE']
labels=pd['KIDNAP & ABDUCTION']


# In[52]:


SLOPE,INTERCEPT,r,p,std_err = stats.linregress(features,labels)


# In[53]:


def linefunc(x):
    return SLOPE*x + INTERCEPT


# In[54]:


lineY = list(map(linefunc,features))
print(lineY)


# In[55]:


plt.scatter(features,labels)
plt.plot(features,lineY)
plt.show()


# In[56]:


case_2=linefunc(6)


# In[57]:


case_2


# In[68]:


reg = np.polyfit(x=pd['RAPE'], y=pd['KIDNAP & ABDUCTION'], deg = 1)
reg


# #### regression line of dowry deaths on cruelty by husband or his relatives(y=a+bx)

# In[58]:


features = pd['DOWRY DEATHS']
labels = pd['CRUETLY BY HUSBAND OR HIS RELATIVES']


# In[59]:


SLOPE,INTERCEPT,r,p,std_err = stats.linregress(features,labels)


# In[60]:


def linefunc(x):
    return SLOPE*x + INTERCEPT


# In[61]:


lineY = list(map(linefunc,features))
print(lineY)


# In[62]:


plt.scatter(features,labels)
plt.plot(features,lineY)
plt.show()


# In[63]:


linefunc(9)


# In[78]:


reg = np.polyfit(x=pd['DOWRY DEATHS'], y=pd['CRUETLY BY HUSBAND OR HIS RELATIVES'], deg = 1)
reg


# #### regression line of human trafficking and imortal trafficking(y=a+bx) 

# In[69]:


Y_VARIABLE=pd['HUMAN TRAFFICKING']
X_VARIABLE=pd['IMORTAL TRAFFICKING']


# In[71]:


SLOPE,INTERCEPT,r,p,std_err = stats.linregress(X_VARIABLE,Y_VARIABLE)


# In[72]:


def linefunc(x):
    return SLOPE*x + INTERCEPT


# In[73]:


lineY = list(map(linefunc,Y_VARIABLE))
print(lineY)


# In[74]:


plt.scatter(X_VARIABLE,Y_VARIABLE)
plt.plot(Y_VARIABLE,lineY)
plt.show()


# In[79]:


reg = np.polyfit(x=pd['HUMAN TRAFFICKING'], y=pd['IMORTAL TRAFFICKING'], deg = 1)
reg


# #### regression line of sexual harrasment on assualt on modesty(y=a+bx) 

# In[81]:


Y_VARIABLE=pd['SEXUAL HARRASSMENT']
X_VARIABLE=pd['ASSAULT ON MODESTY']


# In[82]:


SLOPE,INTERCEPT,r,p,std_err = stats.linregress(X_VARIABLE,Y_VARIABLE)


# In[83]:


def linefunc(x):
    return SLOPE*x + INTERCEPT


# In[84]:


lineY = list(map(linefunc,Y_VARIABLE))
print(lineY)


# In[85]:


plt.scatter(X_VARIABLE,Y_VARIABLE)
plt.plot(Y_VARIABLE,lineY)
plt.show()


# In[87]:


reg = np.polyfit(x=pd['SEXUAL HARRASSMENT'], y=pd['ASSAULT ON MODESTY'], deg = 1)
reg

#graphs to showcase crimes against women in different states in india
# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


import numpy as np
import pandas as pd
import os
os.chdir('G:/SET PROJECT/crime rate prediction/dataset')


# In[13]:


df=pd.read_csv('crime head(2011-2021).csv')
df


# In[14]:


df.head()


# In[15]:


years_title=[str(i) for i in range(2011,2022)]


# In[16]:


STATES_IN_INDIA=df['STATES'].unique()
STATES_IN_INDIA=STATES_IN_INDIA[:-4]

STATES_IN_INDIA


# In[17]:


TYPES_OF_CRIMES=df['CRIME HEAD'].unique()
TYPES_OF_CRIMES=TYPES_OF_CRIMES[:-1]

TYPES_OF_CRIMES


# In[20]:


for state in STATES_IN_INDIA:
    fig = plt.figure(figsize=(18,18), dpi=80, facecolor='w', edgecolor='k')
    plt.title(state)
    plt.xlabel('YEARS')
    plt.ylabel('NO. OF CASES')
    for case in TYPES_OF_CRIMES:
        temp_df = df[(df['STATES'] == state) & (df['CRIME HEAD'] == case)]
        N_crimes = [temp_df[c].values[0] for c in years_title]
        plt.plot(years_title,N_crimes)
        plt.legend(TYPES_OF_CRIMES)
    


# # TOTAL CRIMES AGAINST WOMEN STATE WISE

# In[22]:


fig = plt.figure(figsize = (20,10) , dpi = 80 , facecolor = 'w', edgecolor = 'k')
plt.title('TOTAL CRIMES YEAR WISE')
plt.xlabel('YEARS')
plt.ylabel('NO. OF CASES')
for state in STATES_IN_INDIA:
        temp_df = df[(df['STATES'] == state) & (df['CRIME HEAD'] == 'TOTAL CRIMES AGAINST WOMEN')]
        N_crimes = [temp_df[c].values[0] for c in years_title]
        plt.plot(years_title,N_crimes)
        plt.legend(STATES_IN_INDIA)

















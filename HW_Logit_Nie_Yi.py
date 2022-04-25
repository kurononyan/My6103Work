#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm

# Part I
titanic = dm.api_dsLand('Titanic', 'id')

# Part II
nfl = dm.api_dsLand('nfl2008_fga')
nfl.dropna(inplace=True)

#%% [markdown]

# # Part I  
# Titanic dataset - statsmodels
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | survival | Survived or not | 0 = No, 1 = Yes |  
# | pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |  
# | sex | Gender / Sex |  |  
# | age | Age in years |  |  
# | sibsp | # of siblings / spouses on the Titanic |  |  
# | parch | # of parents / children on the Titanic |  |  
# | ticket | Ticket number (for superstitious ones) |  |  
# | fare | Passenger fare |  |  
# | embarked | Port of Embarkation | C: Cherbourg, Q: Queenstown, S: Southampton  |  
# 
#%%
# ## Question 1  
# With the Titanic dataset, perform some summary visualizations:  
# 
# ### a. Histogram on age. Maybe a stacked histogram on age with male-female as two series if possible
titanic.dropna()
titanic.pivot(columns='sex').age.plot(kind = 'hist', stacked=True)
plt.title("Stacked histogram on age with sex")
plt.xlabel("Age")
plt.show()
# ### b. proportion summary of male-female, survived-dead  
crosstab = pd.crosstab(titanic['survived'], titanic['sex'], margins = True)
print(crosstab)
# ### c. pie chart for “Ticketclass”  
Ticketclass = titanic['pclass'].value_counts()
fig, ax = plt.subplots()
ax.pie(Ticketclass, labels=Ticketclass.index, autopct='%1.1f%%')
plt.title("pie chart for Ticketclass")
plt.legend()
plt.show()
# ### d. A single visualization chart that shows info of survival, age, pclass, and sex.  
sexcolors = np.where(titanic['pclass']==1,'r','-') 
sexcolors[titanic['pclass']==2] = 'b'
sexcolors[titanic['pclass']==3] = 'g'

ax1 = titanic[titanic.sex=='male'].plot(x="age", y="survived", kind="scatter", color=sexcolors[titanic.sex=='male'], marker='^', s=7, label='male')
titanic[titanic.sex=='female'].plot(x="age", y="survived", kind="scatter", color=sexcolors[titanic.sex=='female'], marker='+', s=7, label='female', ax = ax1)
plt.title("Age vs Survival")
plt.show()

#%%
# ## Question 2  
# Build a logistic regression model for survival using the statsmodels library. As we did before, include the features that you find plausible. Make sure categorical variables are use properly. If the coefficient(s) turns out insignificant, drop it and re-build.  
import statsmodels.api as sm
from statsmodels.formula.api import glm
Logit = glm(formula='survived~C(pclass)+age+C(sex)+sibsp+parch', data=titanic, family=sm.families.Binomial())
LogitFit = Logit.fit()
print(LogitFit.summary())

# When alpha = 0.05, the p-value of "parch" is 0.840. Thus we fail to reject the null hypothesis that the coefficient of "parch" is 0. 
#%%
# Re-build the model. 
Logit_re = glm(formula='survived~C(pclass)+age+C(sex)+sibsp', data=titanic, family=sm.families.Binomial())
LogitFit_re = Logit_re.fit()
print(LogitFit_re.summary())

#%%
# ## Question 3  
# Interpret your result. What are the factors and how do they affect the chance of survival (or the survival odds ratio)? What is the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, 3 parents/children on the trip? Use whatever variables that are relevant in your model.  

# Using pclass =1 as the base line, when the pclass =2, the odds ratio of survived becomes exp(-0.9510) times. When the pclass =3, the odds ratio of survived becomes exp(-2.1575) times.
# Using female as the base line, the odds ratio of male survived becomes exp(-2.7393) times.
# 1 degree increase in age makes odds ratio of survived exp(-0.0181) times as before while other variables stay the same. 
# 1 degree increase in age makes odds ratio of survived exp(-0.2804) times as before while other variables stay the same. 
pred = pd.DataFrame({'pclass':2, 'sex':"female", 'age':30, 'sibsp':0}, pd.Index(range(1)))
print("The predicted probability of survival for a 30-year-old female with a second class ticket, no siblings is %.2f" % LogitFit_re.predict(pred))

#%%
# ## Question 4  
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

titanic['sex'][titanic['sex']=='male'] = 0
titanic['sex'][titanic['sex']=='female'] = 1

xsurvival = titanic[['pclass', 'age', 'sex', 'sibsp']]
ysurvival = titanic['survived']


x_trainSurvival, x_testSurvival, y_trainSurvival, y_testSurvival = train_test_split(xsurvival, ysurvival, random_state=1 )


survivallogit = LogisticRegression()
survivallogit.fit(x_trainSurvival, y_trainSurvival)

def predictcutoff(arr, cutoff):
  arrbool = arr[:,1]>cutoff
  arr= arr[:,1]*arrbool/arr[:,1]
  return arr.astype(int)

test = survivallogit.predict_proba(x_testSurvival)
# a
y_pred3 = predictcutoff(test, 0.3)
print(classification_report(y_testSurvival, y_pred3))
# a) The accuracy of the model is 0.78. The precision for "0" is 0.86, for "1" is 0.70. The recall rate for "0" is 0.73, for "1" is 0.84.
print('----------------------------------------------')
y_pred5 = predictcutoff(test, 0.5)
print(classification_report(y_testSurvival, y_pred5))
# b) The accuracy of the model is 0.79. The precision for "0" is 0.79, for "1" is 0.80. The recall rate for "0" is 0.88, for "1" is 0.68.
print('----------------------------------------------')
y_pred7 = predictcutoff(test, 0.7)
print(classification_report(y_testSurvival, y_pred7))
# c) The accuracy of the model is 0.75. The precision for "0" is 0.70, for "1" is 0.95. The recall rate for "0" is 0.98, for "1" is 0.43.
print('----------------------------------------------')
#%%[markdown]
# # Part II  
# NFL field goal dataset - SciKitLearn
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | AwayTeam | Name of visiting team | |  
# | HomeTeam | Name of home team | |  
# | qtr | quarter | 1, 2, 3, 4 |  
# | min | Time: minutes in the game |  |  
# | sec | Time: seconds in the game |  |  
# | kickteam | Name of kicking team |  |  
# | distance | Distance of the kick, from goal post (yards) |  |  
# | timerem | Time remaining in game (seconds) |  |  
# | GOOD | Whether the kick is good or no good | If not GOOD: |  
# | Missed | If the kick misses the mark | either Missed |  
# | Blocked | If the kick is blocked by the defense | or blocked |  
# 
#%% 
# ## Question 5  
# With the nfl dataset, perform some summary visualizations.  
print(nfl.info())

nfl.pivot(columns='GOOD').timerem.plot(kind = 'hist', stacked=True)
plt.title("Stacked histogram on Time remaining with GOOD")
plt.xlabel("Time remaining in game (seconds)")
plt.show()

qtrclass = nfl['qtr'].value_counts()
fig, ax = plt.subplots()
ax.pie(qtrclass, labels=qtrclass.index, autopct='%1.1f%%')
plt.show()

sns.boxplot(x="GOOD", y="distance", data=nfl)
plt.title("Boxplot for GOOD VS distance")
plt.show()
# %%
# ## Question 6  
# Using the SciKitLearn library, build a logistic regression model overall (not individual team or kicker) to predict the chances of a successful field goal. What variables do you have in your model? 
Logit2 = glm(formula='GOOD~min+sec+distance+timerem', data=nfl, family=sm.families.Binomial())
Logit2Fit = Logit2.fit()
print(Logit2Fit.summary())

# When alpha = 0.05, only the p-value of "distance" is less than 0.05. Thus we only include "distance" in our model using the scikitlearn library. 

#%%
xgood = nfl[['distance']]
ygood = nfl['GOOD']


x_traingood, x_testgood, y_traingood, y_testgood = train_test_split(xgood, ygood, random_state=1 )

goodlogit = LogisticRegression()
goodlogit.fit(x_traingood, y_traingood)
print('The coeficient of distance is',goodlogit.coef_)
print('The intercept of distance is',goodlogit.intercept_)

#%%
# ## Question 7  
# Someone has a feeling that home teams are more relaxed and have a friendly crowd, they should kick better field goals. Use your model to find out if that is subtantiated or not. 
nfl['ifhome'] = 0
boolhome = nfl['HomeTeam']==nfl['kickteam']
nfl['ifhome'] = nfl['ifhome'].add(boolhome)

Logitifhome = glm(formula='GOOD~C(ifhome)', data=nfl, family=sm.families.Binomial())
LogitFitifhome = Logitifhome.fit()
print(LogitFitifhome.summary())
# 
#  When alpha = 0.05, the p-value of "ifhome" is 0.093. Thus we fail to reject the null hypothesis that the coefficient of "ifhome" is 0. 
# %% 
# ## Question 8    
# From what you found, do home teams and road teams have different chances of making a successful field goal? If one does, is that true for all distances, or only with a certain range?

# According to the results of question 7, there is no different of making a successful field goal between two home teams and road teams. We also train models of different teams below, the coefficient of distance in two models are close. 
hometeam = nfl[nfl['HomeTeam']==nfl['kickteam']]
roadteam = nfl[nfl['AwayTeam']==nfl['kickteam']]

xgoodhome = hometeam[['distance']]
ygoodhome = hometeam['GOOD']


x_traingoodhome, x_testgoodhome, y_traingoodhome, y_testgoodhome = train_test_split(xgoodhome, ygoodhome, random_state=1 )

goodhomelogit = LogisticRegression()
goodhomelogit.fit(x_traingoodhome, y_traingoodhome)
print('When being hometeams, the coeficient of distance is',goodhomelogit.coef_)
print('When being hometeams, the intercept of distance is',goodhomelogit.intercept_)
hometeam = nfl[nfl['HomeTeam']==nfl['kickteam']]
roadteam = nfl[nfl['AwayTeam']==nfl['kickteam']]

xgoodroad = roadteam[['distance']]
ygoodroad = roadteam['GOOD']


x_traingoodroad, x_testgoodroad, y_traingoodroad, y_testgoodroad = train_test_split(xgoodroad, ygoodroad, random_state=1 )

goodroadlogit = LogisticRegression()
goodroadlogit.fit(x_traingoodroad, y_traingoodroad)
print('When being roadteams, the coeficient of distance is',goodroadlogit.coef_)
print('When being roadteams, the intercept of distance is',goodroadlogit.intercept_)

# %%
# titanic.dropna()


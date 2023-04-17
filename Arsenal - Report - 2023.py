#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
#from sklearn.metrics import roc_curve, auc_curve
#confussion_matrix,
#X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=126)


# In[2]:


df = pd.read_csv('premierleague.csv', index_col='Unnamed: 0')
df


# In[150]:


df.columns.to_list()


# In[3]:


df.drop(['link_match'], axis=1, inplace=True)


# In[4]:


#seperate goals to create a serie for W/L/D to use it later in the predection:
df[['home_Goals', 'away_Goals']] = df['result_full'].str.split('-', expand = True)
df.head()


# In[5]:


df = df.astype({'home_Goals': int, 'away_Goals': int})


# In[6]:


df[['home_Goals', 'away_Goals']].info()


# In[7]:


df['GoalDifference'] = df['home_Goals'] - df['away_Goals']
df.head()
#note: remove (-) from the 'GoalDifference' column result to avoid miss understanding in the prediction!


# In[8]:


df['points_home'] = 0
df['points_away'] = 0
# idk what is df.index!
for i in df.index:
    if df['home_Goals'][i] > df['away_Goals'][i]:
        df['points_home'][i] += 3
       
    elif df['home_Goals'][i]<df['away_Goals'][i]:
        df['points_away'][i]+=3
      
    else:
        df['points_home'][i]+=1
        df['points_away'][i]+=1
df


# In[9]:


df.sort_values(by = 'date', inplace= True)
df.head(20)


# In[10]:


# create a col to count matches for each season.
matchperday =[]
for i in range(0,4070):
    matchperday.append(i-i+1)

len(matchperday)


# In[11]:


df['matchperday'] = matchperday
df.head()
df['matchperday'].info()


# In[12]:


df.head()


# In[13]:


df_home = df.filter(regex='home', axis=1)
df_home.isna().sum()


# In[14]:


#df_home.dropna(axis=1,inplace=True)


# In[15]:


df_home.isna().sum()


# In[16]:


home= df_home.join([df.season,df.date,df.matchperday])


# In[17]:


home.head()


# In[18]:


hTeams = home.groupby(['season','home_team']).sum()


# In[19]:


hTeams.drop('20/21',axis=0,inplace=True)


# In[20]:


hTeams.reset_index(inplace=True)


# In[21]:


hTeams.rename(columns={'matchperday': 'home_matchperday','home_team':'team_name'}, inplace=True)


# In[22]:


hTeams.set_index('season', inplace=True)


# In[23]:


hTeams


# In[24]:


df_away = df.filter(regex='away', axis=1)


# In[25]:


#df_away.dropna(axis=1,inplace=True)


# In[26]:


df_away.isna().sum()


# In[27]:


away= df_away.join([df.season,df.date,df.matchperday])


# In[28]:


away.head()


# In[29]:


aTeams = away.groupby(['season','away_team']).sum()


# In[30]:


aTeams


# In[31]:


aTeams.drop('20/21',axis=0, inplace=True)


# In[32]:


aTeams.reset_index(inplace=True)


# In[33]:


aTeams.set_index('season', inplace=True)


# In[34]:


aTeams.rename(columns={'matchperday': 'away_matchperday','away_team':'team_name'}, inplace=True)


# In[35]:


aTeams.drop(columns ='team_name', axis=1,inplace=True)


# In[36]:


aTeams


# In[37]:


#aTeams.drop('season', axis=1)


# In[38]:


aTeams.info()


# In[39]:


hTeams.info()


# In[40]:


teams_df = pd.concat([hTeams,aTeams], axis=1)


# In[41]:


teams_df['totalmatchperday']= teams_df['away_matchperday'] + teams_df['home_matchperday']


# In[42]:


teams_df['total_points']= teams_df['points_home'] + teams_df['points_away']


# In[43]:


teams_df['total_goals_scored']= teams_df['home_Goals'] + teams_df['away_Goals']


# In[44]:


#it needs another cols above
#teams_df['total_goals_conceded']= teams_df['away_matchperday'] + teams_df['home_matchperday']


# In[45]:


teams_df['total_tackles']= teams_df['home_tackles'] + teams_df['away_tackles']


# In[46]:


teams_df['total_touches']= teams_df['home_touches'] + teams_df['away_touches']


# In[47]:


teams_df['total_shots']= teams_df['home_shots'] + teams_df['away_shots']


# In[48]:


teams_df['total_shots_on_target']= teams_df['home_shots_on_target'] + teams_df['away_shots_on_target']


# In[49]:


teams_df['total_possesion']= teams_df['home_possession'] + teams_df['away_possession']


# In[50]:


teams_df['total_passes']= teams_df['home_passes'] + teams_df['away_passes']


# In[51]:


teams_df['total_clearences'] = teams_df['home_clearances'] + teams_df['away_clearances']


# In[52]:


teams_df['total_corners'] = teams_df['home_corners'] + teams_df['away_corners']


# In[53]:


teams_df['total_fouls_conceded'] = teams_df['home_fouls_conceded'] + teams_df['away_fouls_conceded']


# In[54]:


teams_df


# In[55]:


justteams = teams_df['team_name']
justteams


# In[56]:


total_df = teams_df.filter(regex='total', axis=1)
total_df


# In[57]:


allteam_charts = pd.concat([justteams, total_df], axis=1)
allteam_charts


# In[58]:


#df[(df['home_team'] == 'Arsenal') | (df['away_team'] == 'Arsenal')]
new_charts = allteam_charts[allteam_charts['team_name']== 'Arsenal']
new_charts


# In[59]:


Arsenal_charts_df = new_charts.groupby('season').sum()
Arsenal_charts_df


# In[60]:


all_df = teams_df.groupby(['season','team_name']).sum()
all_df.sort_values('total_points',ascending=False,inplace=True)
#this is the best performance in the last 10 seasons !
all_df['total_points'].head(30)


# In[61]:


all_df.sort_values(['season', 'total_points'], ascending=[True, False])


# In[62]:


#seperate the data of Arsenal Team :
Arsenal = df[(df['home_team'] == 'Arsenal') | (df['away_team'] == 'Arsenal')]
Arsenal.head(10)


# In[63]:


# if arsenal in home_team and points_home = 3 , then write W, = 1, then write D, = 0 , then write L.
# if arsenal in away_team and points_away = 3, then write W, = 1, then write D, = 0 , then write L.


# In[64]:


df['points_home'] = 0
df['points_away'] = 0
# idk what is df.index!
for i in df.index:
    if df['home_Goals'][i] > df['away_Goals'][i]:
        df['points_home'][i] += 3
       
    elif df['home_Goals'][i]<df['away_Goals'][i]:
        df['points_away'][i]+=3
      
    else:
        df['points_home'][i]+=1
        df['points_away'][i]+=1
df


# In[65]:


Arsenal['points_home_new'] = Arsenal['points_home']/1
Arsenal['points_home_new'].astype('str')


# In[66]:


def apply_condition(row):
    # Access values from columns A and B
    a = row['home_team']
    b = row['points_home']
    c = row['away_team']
    d = row['points_away']
    # Apply conditional logic
    if (a == 'Arsenal') & (b == 3):
        result = 'w'
    elif (a == 'Arsenal') & (b == 1):
        result = 'd'
    elif (a == 'Arsenal') & (b == 0):
        result = 'l'
    elif (c == 'Arsenal') & (d == 3):
        result = 'w'
    elif (c == 'Arsenal') & (d == 1):
        result = 'd'
    else:
        result = 'l'
    return result

# Iterate over rows of the DataFrame using a for loop
for index, row in Arsenal.iterrows():
    # Call the function and store the result in a new column 'C'
    Arsenal.at[index, 'status'] = apply_condition(row)

# Print the updated DataFrame
#print("Updated DataFrame:")
Arsenal.status


# In[67]:


# drop un necessary cols.


# In[68]:


Arsenal.head()


# In[69]:


Arsenal.status


# In[70]:


abc = Arsenal_charts_df.corr()


# In[71]:


abc.columns.tolist()


# In[72]:


piaring_Arsenal_charts_df = Arsenal_charts_df[['total_points', 'total_goals_scored','total_touches','total_clearences','total_tackles']] 


# In[153]:


fig = plt.subplots(figsize=(15,15))
sns.heatmap(abc, annot=True,)


# In[74]:


#according to this heat for arsenal performance in all season we can see that, 
# total points for arsenal in each season was too related to tackles, touches, goals scored and clearnces.


# In[75]:


#charts for these relations:


# In[76]:


Arsenal_charts_df


# In[77]:



#sns.pairplot(piaring_Arsenal_charts_df)
#plt.suptitle('most related')
#plt.show()


# In[78]:


#sns.scatterplot(data=df, x='x', y='y', size='z', marker='h', s=200, legend=False)
#plt.figure(figsize=(5, 5))
#sns.scatterplot(x='total_tackles',y='total_points' ,hue= 'season',size='total_touches',style='total_clearences' ,legend=False, data= Arsenal_charts_df)
#plt.xlabel('total_tackles')
#plt.ylabel('total_points')
#plt.title('Relationship between total_tackles, total_points, per season')
# plt.show()


# In[79]:


plt.figure(figsize=(5, 5))
sns.scatterplot(x='total_tackles',y='total_points' ,hue= 'season', data= Arsenal_charts_df)
plt.xlabel('total_tackles')
plt.ylabel('total_points')
plt.title('Relationship between total_tackles, total_points, per season')
plt.show()


# In[80]:


plt.figure(figsize=(5, 5))
sns.scatterplot(x='total_goals_scored',y='total_points' ,hue= 'season', data= Arsenal_charts_df)
plt.xlabel('total_goals_scored')
plt.ylabel('total_points')
plt.title('Relationship between total_goals_scored, total_points, per season')
legend = plt.legend(title='seasons',  bbox_to_anchor=(1.05, 1),fontsize=10)
legend.set_title('seasons', prop={'size': 7}) 
plt.show()


# In[81]:


plt.figure(figsize=(5, 5))
sns.scatterplot(x='total_clearences',y='total_points' ,hue= 'season', data= Arsenal_charts_df)
plt.xlabel('total_clearences')
plt.ylabel('total_points')
plt.title('Relationship between total_points, total_clearences, per season')
legend = plt.legend(title='seasons',fontsize=10)
legend.set_title('seasons', prop={'size': 7}) 
plt.show()


# In[82]:


plt.figure(figsize=(5, 5))
sns.scatterplot(x='total_touches',y='total_points' ,hue= 'season', data= Arsenal_charts_df)
plt.xlabel('total_touches')
plt.ylabel('total_points')
plt.title('Relationship between total_touches, total_points, per season')
legend = plt.legend(title='seasons',fontsize=10)
legend.set_title('seasons', prop={'size': 7}) 
plt.show()


# In[83]:


##charts about sub important relations:
['totalmatchperday',
 'total_points',
 'total_goals_scored',
 'total_tackles',
 'total_touches',
 'total_shots',
 'total_shots_on_target',
 'total_possesion',
 'total_passes',
 'total_clearences',
 'total_corners',
 'total_fouls_conceded']


# In[84]:


plt.figure(figsize=(5, 5))
sns.scatterplot(x='total_possesion',y='total_shots' ,size = 'total_goals_scored',hue= 'season', data= Arsenal_charts_df)
plt.xlabel('total_touches')
plt.ylabel('total_points')
plt.title('Relationship between total_touches, total_points, per season')
legend = plt.legend(title='seasons',  bbox_to_anchor=(1.05, 1),fontsize=10)
legend.set_title('seasons', prop={'size': 7}) 
plt.show()


# In[155]:


plt.figure(figsize=(5, 5))
sns.scatterplot(x='total_clearences',y='total_tackles' ,size = 'total_points',hue= 'season', data= Arsenal_charts_df)
plt.xlabel('total_clearences')
plt.ylabel('total_tackles')
plt.title('Relationship between total_clearences, total_tackles, per season')
legend = plt.legend(title='seasons',  bbox_to_anchor=(1.05, 1),fontsize=10)
legend.set_title('seasons', prop={'size': 7}) 
plt.show()


# In[86]:


plt.figure(figsize=(5, 5))
sns.scatterplot(x='total_possesion',y='total_passes' ,size = 'total_points',hue= 'season', data= Arsenal_charts_df)
plt.xlabel('total_possesion')
plt.ylabel('total_passes')
#plt.title('Relationship between total_touches, total_points, per season')
legend = plt.legend(title='seasons',  bbox_to_anchor=(1.05, 1),fontsize=10)
legend.set_title('seasons', prop={'size': 7}) 
plt.show()
#,style='total_shots_on_target'


# In[87]:


### teams performances:
allteam_charts


# In[88]:


at = all_df.sort_values(['season', 'total_points'], ascending=[True, False])


# In[89]:


aaa = all_df['total_points'].head(20)
bbb = aaa.reset_index()
bbb


# In[90]:


sns.barplot(data = bbb,x='season',y='total_points',hue='team_name')
#sns.lineplot
plt.xlabel('season')
plt.ylabel('total_points')
#plt.title('Relationship between total_touches, total_points, per season')
legend = plt.legend(title='seasons',  bbox_to_anchor=(1.05, 1),fontsize=10)
legend.set_title('seasons', prop={'size': 7}) 
plt.show()


# In[91]:


#best season for arsenal is 13/14
#bet teams against arsenal are: city 17/18, liverpool 19/20, chelsea 16/17, man u 11/12, tottenham 16/17


# In[92]:


at.reset_index()


# In[93]:


at.columns.tolist()


# In[94]:


col_to_drop = ['home_clearances',
 'home_corners',
 'home_fouls_conceded',
 'home_offsides',
 'home_passes',
 'home_possession',
 'home_red_cards',
 'home_shots',
 'home_shots_on_target',
 'home_tackles',
 'home_touches',
 'home_yellow_cards',
 'goal_home_ft',
 'goal_home_ht',
 'clearances_avg_home',
 'corners_avg_home',
 'fouls_conceded_avg_home',
 'offsides_avg_home',
 'passes_avg_home',
 'possession_avg_home',
 'red_cards_avg_home',
 'shots_avg_home',
 'shots_on_target_avg_home',
 'tackles_avg_home',
 'touches_avg_home',
 'yellow_cards_avg_home',
 'goals_scored_ft_avg_home',
 'goals_conced_ft_avg_home',
 'sg_match_ft_acum_home',
 'goals_scored_ht_avg_home',
 'goals_conced_ht_avg_home',
 'sg_match_ht_acum_home',
 'performance_acum_home',
 'home_Goals',
 'points_home',
 'home_matchperday',
 'away_clearances',
 'away_corners',
 'away_fouls_conceded',
 'away_offsides',
 'away_passes',
 'away_possession',
 'away_red_cards',
 'away_shots',
 'away_shots_on_target',
 'away_tackles',
 'away_touches',
 'away_yellow_cards',
 'goal_away_ft',
 'goal_away_ht',
 'clearances_avg_away',
 'corners_avg_away',
 'fouls_conceded_avg_away',
 'offsides_avg_away',
 'passes_avg_away',
 'possession_avg_away',
 'red_cards_avg_away',
 'shots_avg_away',
 'shots_on_target_avg_away',
 'tackles_avg_away',
 'touches_avg_away',
 'yellow_cards_avg_away',
 'goals_scored_ft_avg_away',
 'goals_conced_ft_avg_away',
 'sg_match_ft_acum_away',
 'goals_scored_ht_avg_away',
 'goals_conced_ht_avg_away',
 'sg_match_ht_acum_away',
 'performance_acum_away',
 'away_Goals',
 'points_away',
 'away_matchperday',]


# In[95]:


atd = at.drop(col_to_drop, axis=1)
atdri = atd.reset_index()


# In[96]:


atdri


# In[97]:


#bet teams against arsenal are: city 17/18, liverpool 19/20, chelsea 16/17, man u 11/12, tottenham 16/17

#best season for arsenal is 13/14

#worst Arsenal is


# In[98]:


b_mcity = atdri[(atdri['season'] == '17/18')]
best_mcity = b_mcity[(b_mcity['team_name'] == 'Manchester City')]


# In[99]:


best_mcity


# In[100]:


b_liverpool = atdri[(atdri['season'] == '19/20')]
best_liverpool = b_liverpool[(b_liverpool['team_name'] == 'Liverpool')]


# In[101]:


best_liverpool


# In[102]:


b_Chelsea = atdri[(atdri['season'] == '16/17')]
best_Chelsea = b_Chelsea[(b_Chelsea['team_name'] == 'Chelsea')]


# In[103]:


best_Chelsea


# In[104]:


b_manu = atdri[(atdri['season'] == '11/12')]
best_manu = b_manu[(b_manu['team_name'] == 'Manchester United')]


# In[105]:


best_manu


# In[106]:


b_tth = atdri[(atdri['season'] == '16/17')]
best_tth = b_tth[(b_tth['team_name'] == 'Tottenham Hotspur')]


# In[107]:


best_tth


# In[108]:


b_aresnal = atdri[(atdri['season'] == '13/14')]
best_aresnal = b_aresnal[(b_aresnal['team_name'] == 'Arsenal')]


# In[109]:


best_aresnal


# In[110]:


#worst arsenal is 19/20
w_aresnal = atdri[(atdri['season'] == '19/20')]
worst_aresnal = w_aresnal[(w_aresnal['team_name'] == 'Arsenal')]


# In[111]:


worst_aresnal


# In[112]:


Best_Teams_vs_Arsenal = pd.concat([best_mcity,best_liverpool,best_Chelsea,best_manu,best_tth,best_aresnal],axis=0)


# In[113]:


Best_Teams_vs_Arsenal.drop('totalmatchperday', axis=1, inplace=True)


# In[114]:


Best_Teams_vs_Arsenal


# In[115]:


sns.set(style="whitegrid")
plt.figure(figsize=(4,4))
sns.scatterplot(x='total_goals_scored' ,y="total_points", hue="team_name",size='total_tackles', data=Best_Teams_vs_Arsenal)
plt.xlabel('total_goals_scored')
plt.ylabel('total_points')
plt.title('Comparison of Arsenal with Other Teams')
plt.legend(title='Relationship',  bbox_to_anchor=(1.05, 1))
plt.show()


# In[116]:


sns.set(style="whitegrid")
plt.figure(figsize=(4,4))
sns.scatterplot(x='total_touches' ,y="total_points", hue="team_name",size='total_shots', data=Best_Teams_vs_Arsenal)
plt.xlabel('total_touches')
plt.ylabel('total_points')
plt.title('Comparison of Arsenal with Other Teams')
plt.legend(title='Relationship',  bbox_to_anchor=(1.05, 1))
plt.show()


# In[117]:


sns.set(style="whitegrid")
plt.figure(figsize=(4,4))
sns.scatterplot(x='total_shots_on_target' ,y="total_points", hue="team_name",size='total_possesion', data=Best_Teams_vs_Arsenal)
plt.xlabel('total_shots_on_target')
plt.ylabel('total_points')
plt.title('Comparison of Arsenal with Other Teams')
plt.legend(title='Relationship',  bbox_to_anchor=(1.05, 1))
plt.show()


# In[118]:


sns.set(style="whitegrid")
plt.figure(figsize=(4,4))
sns.scatterplot(x='total_passes' ,y="total_points", hue="team_name",size='total_corners', data=Best_Teams_vs_Arsenal)
plt.xlabel('total_passes')
plt.ylabel('total_points')
plt.title('Comparison of Arsenal with Other Teams')
plt.legend(title='Relationship',  bbox_to_anchor=(1.05, 1))
plt.show()


# In[119]:


sns.set(style="whitegrid")
plt.figure(figsize=(4,4))
sns.scatterplot(x='total_clearences' ,y="total_points", hue="team_name",size='total_fouls_conceded', data=Best_Teams_vs_Arsenal)
plt.xlabel('total_clearences')
plt.ylabel('total_points')
plt.title('Comparison of Arsenal with Other Teams')
plt.legend(title='Relationship',  bbox_to_anchor=(1.05, 1))
plt.show()


# In[120]:


Worst_Arsenal_vs_Best_Arsenal = pd.concat([best_aresnal,worst_aresnal],axis=0)
Worst_Arsenal_vs_Best_Arsenal.drop('totalmatchperday', axis=1, inplace=True)


# In[121]:


#worst arsenal is 19/20

Worst_Arsenal_vs_Best_Arsenal


# In[122]:


sns.set(style="whitegrid")
plt.figure(figsize=(4,4))
sns.scatterplot(x='total_goals_scored' ,y="total_points", hue="team_name",size='total_clearences', data=Worst_Arsenal_vs_Best_Arsenal)
plt.xlabel('total_goals_scored')
plt.ylabel('total_points')
plt.title('Comparison of best Arsenal with worst Arsenal')
plt.legend(title='Relationship',  bbox_to_anchor=(1.05, 1))
plt.show()


# In[123]:


#power bi report creation from the total data concept to show how far Arsenal is from the best place
#ans so far arsenal has to improve their data thorugh getting good players in each position who can fill up the shortage in these points
#also it is necessary to have the players ratings in each match in each season to compare that too.
#and also necessary to have the data for the ticket sales which will indicates the satisfication from the team per season.


# In[124]:


Best_Teams_vs_Arsenal.to_csv('Premium.L_Teams_charts_of_totals.csv', index=False)
file_path = '/Users/wolf/Desktop/Best_Teams_vs_Arsenal.csv'  
# Replace with your desktop path
Best_Teams_vs_Arsenal.to_csv(file_path, index=False) 


# # machine learning process:

# In[125]:


# create data = x
# target = y
#get_dummies for Target col


# In[126]:


status_dummies = pd.get_dummies(Arsenal['status'])
status_dummies


# In[127]:


ML_Arsenal = pd.concat([Arsenal,status_dummies], axis=1)
ML_Arsenal


# In[128]:


Totals = ML_Arsenal.groupby(['season'])[['w', 'd', 'l']].sum()
Totals


# In[129]:


ML_Arsenal.columns.tolist()


# In[130]:


# List of column names to drop
columns_to_drop = ['clearances_avg_H','result_ht','sg_match_ht',
 'corners_avg_H',
 'fouls_conceded_avg_H',
 'offsides_avg_H',
 'passes_avg_H',
 'possession_avg_H',
 'red_cards_avg_H',
 'shots_avg_H',
 'shots_on_target_avg_H',
 'tackles_avg_H',
 'touches_avg_H',
 'yellow_cards_avg_H',
 'goals_scored_ft_avg_H',
 'goals_conced_ft_avg_H',
 'goals_scored_ht_avg_H',
 'goals_conced_ht_avg_H',
 'sg_match_ht_acum_H', 'points_home_new','result_full','season','date','home_team','away_team',
 'GoalDifference','status','d','l',
]

# Drop multiple columns
ML_Arsenal = ML_Arsenal.drop(columns_to_drop, axis=1)


# In[131]:


ML_Arsenal.head()


# In[132]:


X = ML_Arsenal.drop(columns=['w'],axis=1)
Y = ML_Arsenal.iloc[:,-1:]
X


# In[133]:


X.fillna(value = 0, axis=0, inplace = True)


# In[134]:


X


# In[135]:


Y


# In[136]:


# train model & sc data 


# In[137]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[138]:


sc = StandardScaler()


# In[139]:


X_train_scaled = sc.fit_transform(X_train)
X_test_scaled  = sc.fit_transform(X_test)


# In[140]:


#create the algo
#test the effeiciency and test the best algorithm!


# In[141]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[142]:


svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Predict using the trained SVM model
y_pred = svm.predict(X_test_scaled)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[143]:


classification_report(y_test, y_pred)
report = classification_report(y_test, y_pred)
# Print the classification report

print(report)


# In[144]:


# Define the hyperparameter grid
param_grid = { 
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': [0.001, 0.1, 0.5]
}

# Create an SVM classifier
svm = SVC()

# Perform hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Train the SVM model with the best hyperparameters
svm_best = SVC(**best_params)
svm_best.fit(X_train_scaled, y_train)

# Predict using the trained SVM model
y_pred = svm_best.predict(X_test_scaled)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print('best_params',best_params)


# In[145]:


classification_report(y_test, y_pred)
report = classification_report(y_test, y_pred)
# Print the classification report
print(report)


# In[ ]:





# In[146]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

# Create a Linear Regression model
lr = LogisticRegression()

# Train the Linear Regression model
lr.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = lr.predict(X_test_scaled)

# Evaluate the performance of the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

accuracy = lr.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)
print(confusion_matrix(y_test, y_pred))


# In[ ]:


lr = LogisticRegression(random_state=123) 
parametres = {'C': np.linspace(0.05, 1, 20), 
              'penalty' : ['l1', 'l2', 'elasticnet', 'none'], 
              'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid_lr = GridSearchCV(estimator=lr, param_grid=parametres)


grid_lr.fit(X_train_scaled, y_train)
print("Best Parameters : {}".format(grid_lr.best_params_))
print("score of lr : {}".format(grid_lr.score(X_train_scaled, y_train)))

y_pred_lr = grid_lr.predict(X_test) 


# In[ ]:


classification_report(y_test, y_pred_lr)
report = classification_report(y_test, y_pred_lr)
# Print the classification report
print(report)


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=5)

# Train the KNeighborsRegressor model
knn.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test_scaled)

# Evaluate the performance of the model using Mean Squared Error (MSE)
mse3 = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse3)


accuracy = knn.score(y_pred, y_test)
print("Accuracy:", accuracy)


# In[ ]:


classification_report(y_test, y_pred)
report = classification_report(y_test, y_pred)
# Print the classification report
print(report)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#create reports to show the analysis work. to use its concepts in power bi, and the report.


# In[ ]:


####


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





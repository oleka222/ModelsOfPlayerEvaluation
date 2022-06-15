# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:23:13 2021

@author: aleks
"""
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from socceraction.vaep.formula import value

df_games = pd.read_hdf('spadl.h5', key='games')

dfs_features = []
for i, game in df_games.iterrows():
    game_id = game['game_id']
    df_features = pd.read_hdf('features.h5', key=f'game_{game_id}')
    df_features['game_id'] = game_id
    dfs_features.append(df_features)
df_features = pd.concat(dfs_features).reset_index(drop=True)

dfs_labels = []
for i, game in df_games.iterrows():
    game_id = game['game_id']  
    df_labels = pd.read_hdf('labels.h5', key=f'game_{game_id}')
    df_labels['game_id'] = game_id
    dfs_labels.append(df_labels)
df_labels = pd.concat(dfs_labels).reset_index(drop=True)

df_games_train = df_games[df_games["competition_id"] != 28]
df_games_test = df_games[df_games["competition_id"] == 28]

X_train = df_features[df_features['game_id'].isin(df_games_train["game_id"])]
X_test = df_features[df_features['game_id'].isin(df_games_test["game_id"])]
y_train = df_labels[df_labels['game_id'].isin(df_games_train["game_id"])]
y_test = df_labels[df_labels['game_id'].isin(df_games_test["game_id"])]



features = ['start_x-0', 'start_y-0', 'end_x-0', 'end_y-0','type_id-0',
       'result_id-0', 'bodypart_id-0', 'time_played-0', 
       'start_x-1', 'start_y-1', 'end_x-1', 'end_y-1',
       'type_id-1', 'result_id-1',
       'bodypart_id-1', 'time_played-1', 
       'start_x-2', 'start_y-2',
       'end_x-2', 'end_y-2', 
       'type_id-2', 'result_id-2', 'bodypart_id-2', 'time_played-2',
       'start_distance_to_goal-0', 'start_angle_to_goal-0', 'diff_x-0',
       'diff_y-0', 'distance_covered-0', 
       'end_distance_to_goal-0', 'end_angle_to_goal-0', 
       'start_distance_to_goal-1', 'start_angle_to_goal-1',
       'diff_x-1', 'diff_y-1', 'distance_covered-1', 'end_distance_to_goal-1', 'end_angle_to_goal-1',
       'start_distance_to_goal-2',
       'start_angle_to_goal-2', 'diff_x-2', 'diff_y-2', 'distance_covered-2',
       'end_distance_to_goal-2',
       'end_angle_to_goal-2']



scoring_model = XGBClassifier(n_estimators=100, max_depth=4, enable_categroical = True, n_jobs = -1)
scoring_model.fit(X_train[features], y_train['scores'])
plot_importance(scoring_model)

scoring_probability = scoring_model.predict_proba(X_test[features])
cont_predict = scoring_probability[:, 1]

from sklearn.calibration import calibration_curve
c1 = calibration_curve(y_test["scores"], cont_predict, n_bins = 10)


conceding_model = XGBClassifier(n_estimators=100, max_depth=4, enable_categroical = True, n_jobs = -1)
conceding_model.fit(X_train[features], y_train['concedes'])
plot_importance(conceding_model)

conceding_probability = conceding_model.predict_proba(X_test[features])
cont_predict2 = conceding_probability[:, 1]
c2 = calibration_curve(y_test["concedes"], cont_predict2, n_bins = 10)

from sklearn.metrics import brier_score_loss

brier_score_loss(y_test["scores"], cont_predict)
brier_score_loss(y_test["concedes"], cont_predict2)

scoring_predicitons = pd.DataFrame(cont_predict, index = X_test.index, columns = ["scores"])
conceding_predicitons = pd.DataFrame(cont_predict2, index = X_test.index, columns = ["concedes"])

predictions_df = pd.concat([scoring_predicitons, conceding_predicitons], axis=1).reset_index(drop = True)



df_players = pd.read_hdf('spadl.h5', key='players')
df_teams = pd.read_hdf('spadl.h5', key='teams')    



dfs_actions = []
for i, row in df_games_test.iterrows():
    game_id = row['game_id']
    with pd.HDFStore('spadl.h5') as spadlstore:
        df_actions = spadlstore[f'actions/game_{game_id}']
        df_actions = (
            df_actions.merge(spadlstore['actiontypes'], how='left')
            .merge(spadlstore['results'], how='left')
            .merge(spadlstore['bodyparts'], how='left')
            .merge(spadlstore['players'], how='left')
            .merge(spadlstore['teams'], how='left')
            .reset_index()
            .rename(columns={'index': 'action_id'})
        )
    
    dfs_actions.append(df_actions)
df_actions = pd.concat(dfs_actions).reset_index(drop = True)


df_actions_predictions = pd.concat([df_actions, predictions_df], axis=1)
dfs_values = []
for game_id, game_predictions in df_actions_predictions.groupby('game_id'):
    df_values = value(game_predictions, game_predictions['scores'], game_predictions['concedes'])
    
    df_all = pd.concat([game_predictions, df_values], axis=1)
    dfs_values.append(df_all)
    

df_values = pd.concat(dfs_values)
df_values = df_values.sort_values(['game_id', 'period_id', 'time_seconds'])
df_values = df_values.reset_index(drop=True)



ranking = df_values[['player_id', 'team_name', 'short_name', 'vaep_value']].groupby(['player_id', 'team_name', 'short_name'])
ranking = ranking.agg(vaep_count=('vaep_value', 'count'), vaep_mean=('vaep_value', 'mean'), vaep_sum=('vaep_value', 'sum'))
ranking = ranking.sort_values('vaep_sum', ascending=False)
ranking = ranking.reset_index()



df_player_games = pd.read_hdf('spadl.h5', 'player_games')
df_player_games = df_player_games[df_player_games['game_id'].isin(df_games_test['game_id'])]

#adding extra time minutes
df_player_games.loc[(df_player_games["game_id"] == 2058015) & (df_player_games["player_id"].isin([12829, 210044, 69964, 69411, 69400,14771])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058015) & (df_player_games["player_id"].isin([12829, 210044,69964, 69411, 69400,14771]))]["minutes_played"].apply(lambda x: x+26)
df_player_games.loc[(df_player_games["game_id"] == 2058015) & (df_player_games["player_id"].isin([9380,8653,8945,8717,13484,10131,7934,69409,25393,135747,3476,69396,69968,14812,397178,8292])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058015) & (df_player_games["player_id"].isin([9380,8653,8945,8717,13484,10131,7934,69409,25393,135747,3476,69396,69968,14812,397178,8292]))]["minutes_played"].apply(lambda x: x+30)


df_player_games.loc[(df_player_games["game_id"] == 2058012) & (df_player_games["player_id"].isin([14771, 101590])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058012) & (df_player_games["player_id"].isin([14771, 101590]))]["minutes_played"].apply(lambda x: x+25)
df_player_games.loc[(df_player_games["game_id"] == 2058012) & (df_player_games["player_id"].isin([103668,220971,41123,103682,101647,101576,101583,101953,101707,102157,25393,135747,14943,3476,8287,69396,69616,69968,69964,69404])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058012) & (df_player_games["player_id"].isin([103668,220971,41123,103682,101647,101576,101583,101953,101707,102157,25393,135747,14943,3476,8287,69396,69616,69968,69964,69404]))]["minutes_played"].apply(lambda x: x+30)

df_player_games.loc[(df_player_games["game_id"] == 2058009) & (df_player_games["player_id"].isin([3531, 8292,397178])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058009) & (df_player_games["player_id"].isin([3531, 8292,397178]))]["minutes_played"].apply(lambda x: x+24)
df_player_games.loc[(df_player_games["game_id"] == 2058009) & (df_player_games["player_id"].isin([9380,8653,8945,8717,7964,10131,7934,210044,12829,246928,25662,257762,91702,256634,20751,3450,37831,91502,20764])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058009) & (df_player_games["player_id"].isin([9380,8653,8945,8717,7964,10131,7934,210044,12829,246928,25662,257762,91702,256634,20751,3450,37831,91502,20764]))]["minutes_played"].apply(lambda x: x+30)

df_player_games.loc[(df_player_games["game_id"] == 2058005) & (df_player_games["player_id"].isin([241945, 56025,69411,69400])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058005) & (df_player_games["player_id"].isin([241945, 56025,69411,69400]))]["minutes_played"].apply(lambda x: x+27)
df_player_games.loc[(df_player_games["game_id"] == 2058005) & (df_player_games["player_id"].isin([55957,56394,8480,54,55979,56274,20433,405,15080,69409,25393,135747,3476,8287,69396,69616,69404,69964])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058005) & (df_player_games["player_id"].isin([55957,56394,8480,54,55979,56274,20433,405,15080,69409,25393,135747,3476,8287,69396,69616,69404,69964]))]["minutes_played"].apply(lambda x: x+30)

df_player_games.loc[(df_player_games["game_id"] == 2058004) & (df_player_games["player_id"].isin([101953, 70129])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058004) & (df_player_games["player_id"].isin([101953, 70129]))]["minutes_played"].apply(lambda x: x+26)
df_player_games.loc[(df_player_games["game_id"] == 2058004) & (df_player_games["player_id"].isin([103668,41123,257800,103682,101647,101576,101583,101682,4513,101707,7910,3443,3306,3346,3341,3269,3563,3353,4501,3840])), "minutes_played"] = df_player_games[(df_player_games["game_id"] == 2058004) & (df_player_games["player_id"].isin([103668,41123,257800,103682,101647,101576,101583,101682,4513,101707,7910,3443,3306,3346,3341,3269,3563,3353,4501,3840]))]["minutes_played"].apply(lambda x: x+30)


minutes = df_player_games[['player_id', 'minutes_played']]
minutes = minutes.groupby('player_id').sum().reset_index()

per90 = ranking.merge(minutes)
per90['vaep_rating'] = per90['vaep_sum'] * 90 / per90['minutes_played']
per90['actions_p90'] = per90['vaep_count'] * 90 / per90['minutes_played']

per90 = per90[per90['minutes_played']>150]
per90 = per90.sort_values('vaep_rating', ascending=False)

import matplotlib.pyplot as plt
fop1, mpv1 = c1
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(mpv1, fop1, marker='.')
plt.show()

fop2, mpv2 = c2
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(mpv2, fop2, marker='.')
plt.show()
plt.figure(figsize=(20, 10))
x = list(per90['vaep_mean'])
y = list(per90['actions_p90'])
plt.plot(x, y, '.', c='#1C3460', markersize=15)
x_best = list(per90['vaep_mean'][0:10])
y_best = list(per90['actions_p90'][0:10])
names = list(per90['short_name'][0:10])
names = [name.split(".")[-1] for name in names]
plt.plot(x_best, y_best, '.', c='#D62A2E', markersize=15)
for i, txt in enumerate(names):
    plt.annotate(txt, (x[i], y[i] + 2), fontsize=20, horizontalalignment='center')
import numpy as np
best_player = x[0] * y[0]
yi = np.arange(0.1, 140, 0.1)
xi = [best_player / i for i in yi]
plt.plot(xi, yi, '--', c='grey')
plt.title("Quality and quantity of actions", fontsize = 20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, 0.03)
plt.ylim(0, 140)
plt.xlabel('Average VAEP rating per action', labelpad=20, fontsize=20)
plt.ylabel('Total number of actions per 90 minutes', labelpad=20,
            verticalalignment='center', fontsize=20)

plt.show()
"""
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(scoring_model, method='sigmoid', cv=5)
calibrated.fit(X_train[features], y_train['scores'])
# predict probabilities
probs = calibrated.predict_proba(X_test[features])[:, 1]
# reliability diagram
fop3, mpv3 = calibration_curve(y_test['scores'], probs, n_bins=10, normalize=True)
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
plt.plot(mpv3, fop3, marker='.')
plt.show()

calibrated2 = CalibratedClassifierCV(conceding_model, method='sigmoid', cv=5)
calibrated2.fit(X_train[features], y_train['concedes'])
# predict probabilities
probs = calibrated.predict_proba(X_test[features])[:, 1]
# reliability diagram
fop4, mpv4 = calibration_curve(y_test['concedes'], probs, n_bins=10, normalize=True)
# plot perfectly calibrated
plt.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
plt.plot(mpv4, fop4, marker='.')
plt.show()





plt.figure(figsize = (20, 20))
plt.tick_params(axis='both', which='major', labelsize=6)
fig, axs = plt.subplots(2, 2, figsize = (5, 5))
axs[0, 0].plot([0, 1], [0, 1], linestyle='--')
axs[0,0].plot(mpv1, fop1, marker='.')
axs[0, 0].set_title('Scoring model', fontsize = 10)
axs[0, 0].set_xlabel("Predicted probability", fontsize = 8)
axs[0, 0].set_ylabel("True probability", fontsize = 8)
axs[0, 1].plot([0, 1], [0, 1], linestyle='--')
axs[0,1].plot(mpv2, fop2, marker='.')
axs[0, 1].set_title('Conceding model', fontsize = 10)
axs[0, 1].set_xlabel("Predicted probability", fontsize = 8)
axs[0, 1].set_ylabel("True probability", fontsize = 8)

axs[1, 0].plot([0, 1], [0, 1], linestyle='--')


axs[1,0].plot(mpv3, fop3, marker='.')
axs[1, 0].set_title('Scoring model (after calibration)', fontsize = 10)
axs[1, 0].set_xlabel("Predicted probability", fontsize = 8)
axs[1, 0].set_ylabel("True probability", fontsize = 8)
axs[1, 1].plot([0, 1], [0, 1], linestyle='--')
axs[1,1].plot(mpv4, fop4, marker='.')
axs[1, 1].set_title('Conceding model (after calibration)', fontsize = 10)
axs[1, 1].set_xlabel("Predicted probability", fontsize = 8)
axs[1, 1].set_ylabel("True probability", fontsize = 8)
plt.tight_layout()
plt.show()
"""
fig, axs = plt.subplots(2, 1, figsize = (5, 5))
axs[0].hist(scoring_predicitons)
axs[0].set_title("Scoring model")
axs[0].set_xlabel("Scoring probability")
axs[0].set_ylabel("Number of observations")
axs[1].hist(conceding_predicitons)
axs[1].set_title("Conceding model")
axs[1].set_xlabel("Conceding probability")
axs[1].set_ylabel("Number of observations")
plt.tight_layout()





hist_list = ['start_x-0', 'start_y-0', 'end_x-0', 'end_y-0','type_name-0',
       'result_id-0', 'bodypart_name-0', 'time_played-0', 
       'start_distance_to_goal-0', 'start_angle_to_goal-0', 'diff_x-0',
       'diff_y-0', 'distance_covered-0', 
       'end_distance_to_goal-0', 'end_angle_to_goal-0']



for i in hist_list:
    plt.hist(X_train[i], bins = 100)
    plt.title(i)
    plt.xticks(fontsize=12, rotation=90)
    plt.savefig("grafiki_licencjat/train" + i + ".png")
    plt.show()

for i in hist_list:
    plt.hist(X_test[i], bins = 100)
    plt.title(i)
    plt.xticks(fontsize=12, rotation=90)
    plt.savefig("grafiki_licencjat/test" + i + ".png")
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 12:29:07 2021

@author: aleks
"""
import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

def open_file(string):
    with open (string) as f:
        file = json.load(f)
        
    df = pd.DataFrame(file)
    
    
    return df


df1 = open_file('Wyscout/events_England.json')

df2 = open_file('Wyscout/events_England.json')
df3 = open_file('Wyscout/events_Spain.json')
df4 = open_file('Wyscout/events_France.json')
df5 = open_file('Wyscout/events_Italy.json')


def concat_df (df1, df2, df3, df4, df5):
    frames = [df1, df2, df3, df4, df5]
    new_df = pd.concat(frames)
    return new_df

full_df = concat_df(df1, df2, df3, df4, df5)


def drop_headers(df):
   lista = []
   for i, row in df.iterrows():
       for tag in row["tags"]:
           if tag['id'] == 403:
              lista.append(i)
   return df.drop(lista)
    
full_df2 = drop_headers(full_df)
headers_df = full_df.drop(full_df2.index)

def penalty (string):
    with open (string) as f:
        file = json.load(f)
        
    df = pd.DataFrame(file)
    df = df[df['subEventName'].isin(["Penalty"])]
    df = df[df["matchPeriod"] != "P"]
    for i, row in df.iterrows():
        df.at[i,'Goal'] = 0
        for tag in row["tags"]:
            if tag['id'] == 101:
                df.at[i, "Goal"] = 1
                
    return df

def prepare_df_model (df):
    X_df = pd.DataFrame()
    X_df["X"] = df.positions.apply(lambda cell: 100 - cell[0]['x'])
    X_df["X"] = X_df["X"]*105/100
    X_df["C"] = df.positions.apply(lambda cell: cell[0]['y'])
    X_df["C"] = X_df["C"]*68/100
    X_df["Y"] = df.positions.apply(lambda cell: abs(cell[0]['y']-50))
    X_df["Y"] = X_df["Y"]*68 /100
    X_df["Distance"] = np.sqrt(X_df["X"]**2 + X_df["Y"]**2)
    X_df["Angle"] = np.where(np.arctan(7.32 * X_df["X"] /(X_df["X"]**2 + X_df["Y"]**2 - (7.32/2)**2)) > 0, np.arctan(7.32 * X_df["X"] /(X_df["X"]**2 + X_df["Y"]**2 - (7.32/2)**2)), np.arctan(7.32 * X_df["X"] /(X_df["X"]**2 + X_df["Y"]**2 - (7.32/2)**2)) + np.pi)
    for i, row in df.iterrows():
        X_df.at[i,'Goal'] = 0
        for tag in row["tags"]:
            if tag['id'] == 101:
                X_df.at[i, "Goal"] = 1
                
    return X_df

model_df = prepare_df_model(full_df2)
headers_model_df = prepare_df_model(headers_df)

def build_model (df):
    xG_model = smf.glm("Goal ~ Distance + Angle", data = df,  family = sm.families.Binomial()).fit()
    print(xG_model.summary()) 
    return xG_model.params

params = build_model(model_df)
params2 = build_model(headers_model_df)

all_wc = open_file("Wyscout/events_World_Cup.json")
world_cup_shots = drop_headers(all_wc)
world_cup_headers = all_wc.drop(world_cup_shots.index)
world_cup_df = prepare_df_model(world_cup_shots)
world_cup_headers_df = prepare_df_model(world_cup_headers)

def calculate_XG(df, params):
     df["xG"] = 1/(1+np.exp(-(params[0] + (params[1]*df["Distance"]) +(params[2]*df["Angle"]))))
     return df
 
from io import BytesIO
with open('Wyscout/players.json', 'rb') as json_file:
        players = BytesIO(json_file.read()).getvalue().decode('unicode_escape')
    
    
world_cup_df = calculate_XG(world_cup_df, params) 
world_cup_headers_df = calculate_XG(world_cup_headers_df, params2) 
world_cup_shots = pd.concat([world_cup_shots, world_cup_headers])
world_cup_df = pd.concat([world_cup_df, world_cup_headers_df])
df = world_cup_df.join(world_cup_shots)


players_df = pd.read_json(players)
players_df2 = pd.DataFrame()
players_df2["playerId"] = players_df["wyId"]
players_df2["role"] = players_df["role"]
players_df2["shortName"] = players_df["shortName"]

df = df.reset_index().merge(players_df2, how = "inner", on = ["playerId"]).set_index("index")
summary_df = df[["Goal", "xG", "shortName"]]



penalty_wc = penalty("Wyscout/events_World_Cup.json")
penalty_wc = penalty_wc.reset_index().merge(players_df2, how = "inner", on = ["playerId"]).set_index("index")
penalty_wc["xG"] = 0.8
penalty_wc = penalty_wc[["Goal", "xG", "shortName","playerId"]]

summary_df = pd.concat([summary_df, penalty_wc])

xG_sum = summary_df.groupby(["shortName"])["xG"].sum().sort_values(ascending = False)
goals_sum = summary_df.groupby(["shortName"])["Goal"].sum().sort_values(ascending = False)



import matplotlib.pyplot as plt
frame = { 'xG': xG_sum, 'goals': goals_sum}

summary = pd.DataFrame(frame).reset_index()

summary = pd.DataFrame(frame).reset_index()
summary["shortName"] = summary["index"]
df_games = pd.read_hdf('spadl.h5', key='games')
df_games_test = df_games[df_games["competition_id"] == 28]
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
minutes["playerId"] = minutes["player_id"]

minutes = minutes.reset_index().merge(players_df2, how = "inner", on = ["playerId"]).set_index("index")
summary = summary.merge(minutes, how = "inner", on = ["shortName"]).set_index("index")
summary = summary[summary['minutes_played']>150]
summary["xG_90"] = summary["xG"]*90/summary["minutes_played"]
summary["goals_90"] = summary["goals"]*90/summary["minutes_played"]

test = summary.loc[(summary["xG_90"] > 0.48) | (summary["goals_90"] > 0.7)]
not_test = summary.drop(test.index)
summarys = summary.reset_index()
summarys["index"].tolist
fig, ax = plt.subplots(num = 1)
ax.grid(zorder = 1)
ax.scatter(summary["xG_90"], summary["goals_90"])
for i in test.index:
    ax.scatter(summarys.loc[summarys["index"] == i]["xG_90"], summarys.loc[summarys["index"] == i]["goals_90"], edgecolors = "black" ,color = "grey", alpha = 0.3, lw = 0.6, zorder  =3)
    ax.text(summarys.loc[summarys["index"] == i]["xG_90"]-0.05, summarys.loc[summarys["index"] == i]["goals_90"]+0.01, str(i), fontsize = 8, color = "black", zorder = 2)
    
for i in not_test.index:
    ax.scatter(summarys.loc[summarys["index"] == i]["xG_90"], summarys.loc[summarys["index"] == i]["goals_90"],  edgecolors = "black" ,color = "grey", alpha = 0.2, lw = 0.6, zorder = 3)
ax.plot([0, 1], [0, 1],linestyle='dotted', color='red', alpha = 0.5, zorder = 4)
ax.tick_params(axis="x",color="black",length=5, width=1)
ax.tick_params(axis="y",color="black",length=5, width=1)
plt.xlim(0, 1)
plt.ylim(0, 1.3)
ax.set_ylabel("Goals scored per 90 minutes", color='black', fontsize = 14)
ax.set_xlabel("xG per 90 minutes", color='black', fontsize = 14)
plt.ylim((0.00,1.20))
plt.xlim((0.00,1.00))
plt.tight_layout()

pgoal_2d=np.zeros((65,65))
for x in range(65):
    for y in range(65):
        sh=dict()
        a = np.arctan(7.32 *x /(x**2 + abs(y-65/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
        sh['Angle'] = a
        sh['Distance'] = np.sqrt(x**2 + abs(y-65/2)**2)
        sh = pd.DataFrame(sh, index = [x])        
        sh =  calculate_XG(sh, params)
        pgoal_2d[x,y] = sh["xG"]
import FCPython
(fig,ax) = FCPython.createGoalMouth(linecolor = "black")
pos=ax.imshow(pgoal_2d, extent=[-1,65,65,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=1)
fig.colorbar(pos, ax=ax)
ax.set_title('Goal probability')
plt.xlim((0,66))
plt.ylim((-3,35))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
fig.savefig("scoring_prob2.png", dpi=None, bbox_inches="tight")  

pgoal_2d=np.zeros((65,65))
for x in range(65):
    for y in range(65):
        sh=dict()
        a = np.arctan(7.32 *x /(x**2 + abs(y-65/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
        sh['Angle'] = a
        sh['Distance'] = np.sqrt(x**2 + abs(y-65/2)**2)
        sh = pd.DataFrame(sh, index = [x])        
        sh =  calculate_XG(sh, params2)
        pgoal_2d[x,y] = sh["xG"]
import FCPython
(fig,ax) = FCPython.createGoalMouth(linecolor = "black")
pos=ax.imshow(pgoal_2d, extent=[-1,65,65,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=1)
fig.colorbar(pos, ax=ax)
ax.set_title('Goal probability (header)')
plt.xlim((0,66))
plt.ylim((-3,35))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
fig.savefig("scoring_prob_head2.png", dpi=None, bbox_inches="tight")  




plt.hist(model_df["Distance"], bins = 100)
plt.title("Shots - distance")
plt.savefig("grafiki_licencjat/xgshots_dist.png")
plt.show()
plt.hist(model_df["Angle"], bins = 100)
plt.title("Shots - angle")
plt.savefig("grafiki_licencjat/xgshots_ang.png")

plt.show()
plt.hist(headers_model_df["Distance"], bins = 100)
plt.title("Headers - distance")
plt.savefig("grafiki_licencjat/xghead_dist.png")

plt.show()
plt.hist(headers_model_df["Angle"], bins = 100)
plt.title("Headers - angle")
plt.savefig("grafiki_licencjat/xghead_ang.png")

plt.show()
plt.hist(world_cup_df["Distance"], bins = 100)
plt.title("Shots - distance")
plt.savefig("grafiki_licencjat/xgshots_dist_test.png")

plt.show()
plt.hist(world_cup_df["Angle"], bins = 100)
plt.title("Shots - angle")
plt.savefig("grafiki_licencjat/xgshots_ang_test.png")

plt.show()
plt.hist(world_cup_headers_df["Distance"], bins = 100)
plt.title("Headers - distance")
plt.savefig("grafiki_licencjat/xghead_dist_test.png")

plt.show()
plt.hist(world_cup_headers_df["Angle"], bins = 100)
plt.title("Headers - angle")
plt.savefig("grafiki_licencjat/xghead_ang_test.png")

plt.show()
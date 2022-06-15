# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:12:20 2021

@author: aleks
"""
import pandas as pd
import numpy as np
import json

with open ("Wyscout/events_England.json") as f:
     file = json.load(f)
     
df = pd.DataFrame(file)
pd.unique(df["eventName"])

#nie mogę wcziąć wybić bramkarskich bo są zarejestrowane jako 0,0
xT_df = df[df['subEventName'].isin(['Shot', 'Free kick shot', 'Simple pass', 'High pass', 'Throw in', 'Head pass', 'Smart pass','Cross', 'Free kick cross','Corner'])]


xT_df["x0"] = xT_df.positions.apply(lambda cell: cell[0]['x']) * 105/100
xT_df["y0"] = xT_df.positions.apply(lambda cell: cell[0]['y']) * 68/100

move_df = xT_df[xT_df['subEventName'].isin(['Simple pass', 'High pass', 'Throw in', 'Head pass', 'Smart pass','Cross', 'Goal kick','Free kick cross','Corner'])]

move_df['x1'] = move_df.positions.apply(lambda cell: cell[1]['x']) * 105/100
move_df['y1'] = move_df.positions.apply(lambda cell: cell[1]['y']) * 68/100


shot_df = xT_df[xT_df['subEventName'].isin(['Shot', 'Free kick shot'])]
shot_df['x1'] = "S"
shot_df['y1'] = "S"

xT_df = pd.concat([move_df, shot_df])

for i, row in xT_df.iterrows():
    for j in range(0, 16):
        if row["y0"] < (1/12)*68 and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 1 +12*j
        if row["y0"] >= (1/12)*68 and row["y0"] < (2/12)*68 and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 2 +12*j
        if row["y0"] >= (2/12)*68 and row["y0"] < (3/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 3 +12*j
        if row["y0"] >= (3/12)*68  and row["y0"] < (4/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 4 +12*j
        if row["y0"] >= (4/12)*68  and row["y0"] < (5/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 5 +12*j
        if row["y0"] >=(5/12)*68  and row["y0"] < (6/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 6 +12*j
        if row["y0"] >= (6/12)*68  and row["y0"] < (7/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 7 +12*j
        if row["y0"] >= (7/12)*68  and row["y0"] < (8/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 8 +12*j
        if row["y0"] >= (8/12)*68  and row["y0"] < (9/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 9 +12*j
        if row["y0"] >= (9/12)*68  and row["y0"] < (10/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 10 +12*j
        if row["y0"] >= (10/12)*68  and row["y0"] < (11/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 11 +12*j
        if row["y0"] >= (11/12)*68   and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 12 +12*j
        
        
        
        
for i, row in xT_df.iterrows():
    if row["x1"] != "S":
        for j in range(0, 16):
            if row["y1"] < (1/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 1 +12*j
            if row["y1"] >= (1/12)*68  and row["y1"] < (2/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 2 +12*j
            if row["y1"] >= (2/12)*68  and row["y1"] < (3/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 3 +12*j
            if row["y1"] >= (3/12)*68  and row["y1"] < (4/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 4 +12*j
            if row["y1"] >= (4/12)*68  and row["y1"] < (5/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 5 +12*j
            if row["y1"] >=(5/12)*68  and row["y1"] < (6/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 6 +12*j
            if row["y1"] >= (6/12)*68  and row["y1"] < (7/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 7 +12*j
            if row["y1"] >= (7/12)*68  and row["y1"] < (8/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 8 +12*j
            if row["y1"] >= (8/12)*68  and row["y1"] < (9/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 9 +12*j
            if row["y1"] >= (9/12)*68  and row["y1"] < (10/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 10 +12*j
            if row["y1"] >= (10/12)*68  and row["y1"] < (11/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 11 +12*j
            if row["y1"] >= (11/12)*68   and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 12 +12*j
    else:
       xT_df.at[i, "stop"] = 1000
            
delete_df = xT_df.query("x1 == 0 & y1 == 0")
xT_df = xT_df.drop(delete_df.index)


goals_df = xT_df.query("stop == 1000")
xT_df = xT_df.drop(goals_df.index)

count_starts_and_stops = xT_df.groupby(["start", "stop"])["start"].count().unstack(fill_value=0).stack()
count_starts = xT_df.groupby(["start"])["start"].count()

transition_matrix = np.zeros((192,192))

for i in range (1, 187):
    for j in range (1, 193):
        transition_matrix[i-1, j-1] = count_starts_and_stops[i, j]/count_starts[i]

for i in range (188, 193):
    for j in range (1, 193):
        transition_matrix[i-1, j-1] = count_starts_and_stops[i, j]/count_starts[i]
        
        
xT_df = pd.concat([xT_df, goals_df])
shot_starts_and_stops = xT_df.groupby(["start", "stop"])["start"].count().unstack(fill_value=0).stack()
shot_starts = xT_df.groupby(["start"])["start"].count()


shot_probability = pd.DataFrame()

for i in range (1, 193):
    shot_probability.at[i, "prob"] = shot_starts_and_stops[i, 1000]/shot_starts[i]
    
    
    
move_probability = 1 - shot_probability


for i, row in goals_df.iterrows():
        goals_df.at[i,'Goal'] = 0
        for tag in row["tags"]:
            if tag['id'] == 101:
                goals_df.at[i, "Goal"] = 1
                
                
goals_from_area = goals_df.groupby("start")["Goal"].sum()



goal_probability = goals_from_area/shot_starts
goal_probability = goal_probability.fillna(0)



xT = np.zeros(192)
suma = np.zeros(192)
k = 0
while k < 5:
    for i in range(0, 192):
        for j in range (0, 192):
            suma[i] = suma[i] + (transition_matrix[i, j] * xT[j])
        xT[i] = (shot_probability.iloc[i]*goal_probability.iloc[i])+(move_probability.iloc[i]*suma[i])
    suma = np.zeros(192)
    k=k+1

#############################################################

df = pd.read_csv("allsvenskan.csv", sep = ";")
df.loc[~(df["playerId"]==0).all(axis=1)]
#nie mogę wcziąć wybić bramkarskich bo są zarejestrowane jako 0,0
xT_df = df[df['subEventName'].isin(['Shot', 'Free kick shot', 'Simple pass', 'High pass', 'Throw in', 'Head pass', 'Smart pass','Cross', 'Free kick cross','Corner'])]


xT_df["x0"] = xT_df.positions.apply(lambda cell: cell[0]['x']) * 105/100
xT_df["y0"] = xT_df.positions.apply(lambda cell: cell[0]['y']) * 68 /100

move_df = xT_df[xT_df['subEventName'].isin(['Simple pass', 'High pass', 'Throw in', 'Head pass', 'Smart pass','Cross', 'Goal kick','Free kick cross','Corner'])]

#data collecting error dla id 98656, 100297

move_df['x1'] = move_df.positions.apply(lambda cell: cell[1]['x']) * 105/100
move_df['y1'] = move_df.positions.apply(lambda cell: cell[1]['y']) * 68 /100


shot_df = xT_df[xT_df['subEventName'].isin(['Shot', 'Free kick shot'])]
shot_df['x1'] = "S"
shot_df['y1'] = "S"

xT_df = pd.concat([move_df, shot_df])

for i, row in xT_df.iterrows():
    for j in range(0, 16):
        if row["y0"] < (1/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 1 +12*j
        if row["y0"] >= (1/12)*68  and row["y0"] < (2/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 2 +12*j
        if row["y0"] >= (2/12)*68  and row["y0"] < (3/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 3 +12*j
        if row["y0"] >= (3/12)*68  and row["y0"] < (4/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 4 +12*j
        if row["y0"] >= (4/12)*68  and row["y0"] < (5/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 5 +12*j
        if row["y0"] >=(5/12)*68  and row["y0"] < (6/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 6 +12*j
        if row["y0"] >= (6/12)*68  and row["y0"] < (7/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 7 +12*j
        if row["y0"] >= (7/12)*68  and row["y0"] < (8/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 8 +12*j
        if row["y0"] >= (8/12)*68  and row["y0"] < (9/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 9 +12*j
        if row["y0"] >= (9/12)*68  and row["y0"] < (10/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 10 +12*j
        if row["y0"] >= (10/12)*68  and row["y0"] < (11/12)*68  and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 11 +12*j
        if row["y0"] >= (11/12)*68   and row["x0"] < ((j+1)/16)*105 and row["x0"] >= (j/16)*105:
            xT_df.at[i, "start"] = 12 +12*j
        
        
        
        
for i, row in xT_df.iterrows():
    if row["x1"] != "S":
        for j in range(0, 16):
            if row["y1"] < (1/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 1 +12*j
            if row["y1"] >= (1/12)*68  and row["y1"] < (2/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 2 +12*j
            if row["y1"] >= (2/12)*68  and row["y1"] < (3/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 3 +12*j
            if row["y1"] >= (3/12)*68  and row["y1"] < (4/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 4 +12*j
            if row["y1"] >= (4/12)*68  and row["y1"] < (5/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 5 +12*j
            if row["y1"] >=(5/12)*68  and row["y1"] < (6/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 6 +12*j
            if row["y1"] >= (6/12)*68  and row["y1"] < (7/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 7 +12*j
            if row["y1"] >= (7/12)*68  and row["y1"] < (8/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 8 +12*j
            if row["y1"] >= (8/12)*68  and row["y1"] < (9/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 9 +12*j
            if row["y1"] >= (9/12)*68  and row["y1"] < (10/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 10 +12*j
            if row["y1"] >= (10/12)*68  and row["y1"] < (11/12)*68  and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 11 +12*j
            if row["y1"] >= (11/12)*68   and row["x1"] < ((j+1)/16)*105 and row["x1"] >= (j/16)*105:
                xT_df.at[i, "stop"] = 12 +12*j
    else:
       xT_df.at[i, "stop"] = 1000
            
delete_df = xT_df.query("x1 == 0 & y1 == 0")
xT_df = xT_df.drop(delete_df.index)

move_df = xT_df[xT_df['subEventName'].isin(['Simple pass', 'High pass', 'Throw in', 'Head pass', 'Smart pass','Cross', 'Goal kick','Free kick cross','Corner'])]
move_df = move_df.dropna()
for i, row in move_df.iterrows():
    move_df.at[i, "xT_start"] = xT[int(row["start"]-1)]
    move_df.at[i, "xT_stop"] = xT[int(row["stop"] -1)]
    
    
    
for i, thepass in move_df.iterrows():
    for passtags in thepass['tags']:
        if passtags['id']==1801:
            move_df.at[i, 'Accurate'] = 1
        else: 
            move_df.at[i, 'Accurate'] = 0
            
move_df.loc[move_df['Accurate'] == 0, 'xT_stop'] = 0


move_df["difference"] = move_df["xT_stop"] - move_df["xT_start"]


from io import BytesIO
with open('Wyscout/players.json', 'rb') as json_file:
    players = BytesIO(json_file.read()).getvalue().decode('unicode_escape')
players_df = pd.read_json(players)
players_df2 = pd.DataFrame()    
players_df2["playerId"] = players_df["wyId"]
players_df2["role"] = players_df["role"]
players_df2["shortName"] = players_df["shortName"]

df = move_df.reset_index().merge(players_df2, how = "inner", on = ["playerId"]).set_index("index")
summary_df = df[["difference", "shortName", "playerId"]]

xT_sum = summary_df.groupby(["playerId"])["difference"].sum().sort_values(ascending = False).reset_index()

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

players_df2["player_id"] = players_df["wyId"]
minutes_df = minutes.merge(players_df2, how = "inner", on = ["player_id"])
xT_sum["player_id"] = xT_sum["playerId"]
xT_per_90 = minutes.merge(xT_sum, how = "inner", on = ["player_id"])



xT_per_90 = df = players_df2.merge(xT_per_90, how = "inner", on = ["playerId"])
summary_per_df = df[["difference", "shortName", 'minutes_played']]
summary_per_df["per90"] = summary_per_df["difference"]*90/summary_per_df['minutes_played']

summary_per_df = summary_per_df[summary_per_df['minutes_played']>150]
summary_per_df = summary_per_df.sort_values('per90', ascending=False)


pmove_2d=np.zeros((16,12))
for x in range(16):
    for y in range(12):
        pmove_2d[x, y] = move_probability["prob"][(12*(x+1) - y)-1:(12*(x+1) - y)]
    
pshot_2d=np.zeros((16,12))
for x in range(16):
    for y in range(12):
        pshot_2d[x, y] = shot_probability["prob"][(12*(x+1) - y)-1:(12*(x+1) - y)]
        
pgoal_2d=np.zeros((16,12))
for x in range(16):
    for y in range(12):
        pgoal_2d[x, y] = goal_probability[(12*(x+1) - y)]
    
     
        
from FCPython import createPitch
(fig,ax) = createPitch(105,68,'meters','gray')
import matplotlib.pyplot as plt
pos=ax.imshow(pgoal_2d.T, extent=[0,105,68,0], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=1)
ax.set_title('Goal probability')
fig.colorbar(pos, ax=ax)
plt.xlim((0,105))
plt.ylim((0,68))
plt.gca().set_aspect('equal', adjustable='box')
for i in range (1, 16):
    x1, y1 = i*105/16, 0
    x2, y2 = i*105/16, 68
    ax.plot([x1, x2], [y1, y2], color = 'grey', alpha = 0.1)
    
    
for i in range (1, 12):
    x1, y1 = 0, i*68/12
    x2, y2 = 105, i*68/12
    ax.plot([x1, x2], [y1, y2], color = 'grey', alpha = 0.1)
plt.show()
fig.savefig("goal_xT.png", dpi=None, bbox_inches="tight")  


(fig,ax) = createPitch(105,68,'meters','gray')
import matplotlib.pyplot as plt
pos=ax.imshow(pshot_2d.T, extent=[0,105,68,0], aspect='auto',cmap=plt.cm.Blues,vmin=0, vmax=1)
fig.colorbar(pos, ax=ax)
ax.set_title('Shot probability')
plt.xlim((0,105))
plt.ylim((0,68))
plt.gca().set_aspect('equal', adjustable='box')
for i in range (1, 16):
    x1, y1 = i*105/16, 0
    x2, y2 = i*105/16, 68
    ax.plot([x1, x2], [y1, y2], color = 'grey', alpha = 0.1)
    
    
for i in range (1, 12):
    x1, y1 = 0, i*68/12
    x2, y2 = 105, i*68/12
    ax.plot([x1, x2], [y1, y2], color = 'grey', alpha = 0.1)
plt.show()
fig.savefig("shot_probab.png", dpi=None, bbox_inches="tight") 

(fig,ax) = createPitch(105,68,'meters','gray')
import matplotlib.pyplot as plt
pos=ax.imshow(pmove_2d.T, extent=[0,105,68,0], aspect='auto',cmap=plt.cm.Greens,vmin=0, vmax=1)
fig.colorbar(pos, ax=ax)
ax.set_title('Move probability')
plt.xlim((0,105))
plt.ylim((0,68))
plt.gca().set_aspect('equal', adjustable='box')
for i in range (1, 16):
    x1, y1 = i*105/16, 0
    x2, y2 = i*105/16, 68
    ax.plot([x1, x2], [y1, y2], color = 'grey', alpha = 0.1)
    
    
for i in range (1, 12):
    x1, y1 = 0, i*68/12
    x2, y2 = 105, i*68/12
    ax.plot([x1, x2], [y1, y2], color = 'grey', alpha = 0.1)
plt.show()
fig.savefig("move_probab.png", dpi=None, bbox_inches="tight") 

pxT_2d=np.zeros((16,12))
for x in range(16):
    for y in range(12):
        pxT_2d[x, y] = xT[(12*(x+1) - y)-1]
        
        
        
(fig,ax) = createPitch(105,68,'meters','gray')
import matplotlib.pyplot as plt
pos=ax.imshow(pxT_2d.T, extent=[0,105,68,0], aspect='auto',cmap=plt.cm.Purples,vmin=0, vmax=0.5)
fig.colorbar(pos, ax=ax)
ax.set_title('xT value for each zone')
plt.xlim((0,105))
plt.ylim((0,68))
plt.gca().set_aspect('equal', adjustable='box')
for i in range (1, 16):
    x1, y1 = i*105/16, 0
    x2, y2 = i*105/16, 68
    ax.plot([x1, x2], [y1, y2], color = 'grey', alpha = 0.1)
    
    
for i in range (1, 12):
    x1, y1 = 0, i*68/12
    x2, y2 = 105, i*68/12
    ax.plot([x1, x2], [y1, y2], color = 'grey', alpha = 0.1)
plt.show()
fig.savefig("xT_probab.png", dpi=None, bbox_inches="tight") 

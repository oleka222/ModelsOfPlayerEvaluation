# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:01:01 2021

@author: aleks
"""
from io import BytesIO
import pandas as pd  # version 1.0.3
from socceraction.spadl.wyscout import convert_to_spadl

def open_file(filename):
    with open(filename, 'rb') as json_file:
        return BytesIO(json_file.read()).getvalue().decode('unicode_escape')
    
    
teams = open_file('Wyscout/teams.json')  
teams = pd.read_json(teams)
teams.to_hdf('wyscout.h5', key='teams', mode='w')
    
events = open_file('Wyscout/events_Germany.json')    
germany = pd.read_json(events)
events = open_file('Wyscout/events_England.json')    
england = pd.read_json(events)
events = open_file('Wyscout/events_Spain.json')    
spain = pd.read_json(events)
events = open_file('Wyscout/events_Italy.json')    
italy = pd.read_json(events)
events = open_file('Wyscout/events_France.json')    
france = pd.read_json(events)
events = open_file('Wyscout/events_World_Cup.json')    
world_cup = pd.read_json(events)
world_cup = world_cup.loc[world_cup["matchPeriod"] != "P"]
events = pd.concat([germany, england, spain, italy, france, world_cup])
groupby_game = events.groupby(["matchId"], as_index = False)

for i, game in groupby_game:
    game.to_hdf('wyscout.h5', key=f'events/match_{i}', mode='a')


players = open_file('Wyscout/players.json')  
players = pd.read_json(players)
players.to_hdf('wyscout.h5', key='players', mode='a')

matches = open_file('Wyscout/matches_Germany.json')    
germany = pd.read_json(matches)
matches = open_file('Wyscout/matches_England.json')    
england = pd.read_json(matches)
matches = open_file('Wyscout/matches_Spain.json')    
spain = pd.read_json(matches)
matches = open_file('Wyscout/matches_Italy.json')    
italy = pd.read_json(matches)
matches = open_file('Wyscout/matches_France.json')    
france = pd.read_json(matches)
matches = open_file('Wyscout/matches_World_Cup.json')    
world_cup = pd.read_json(matches)


matches = pd.concat([germany, england, spain, italy, france, world_cup])
matches.to_hdf('wyscout.h5', key='matches', mode='a')

#convert to spadl
convert_to_spadl('wyscout.h5', 'spadl.h5')






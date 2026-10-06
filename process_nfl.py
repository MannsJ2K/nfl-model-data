"""Produce one validated atomic JSON feed for the existing NFL model."""
import argparse,datetime,json,os,pathlib
import pandas as pd
import numpy as np
TEAMS='ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LAC LAR LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS'.split()
COLS=['season','season_type','game_id','game_date','play_id','desc','posteam','defteam','home_team','away_team','home_score','away_score','play_type','epa','qb_kneel','qb_spike','qb_dropback','rush','yards_gained','sack']
def load(season,local=None):
 source=local or f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz'
 d=pd.read_csv(source,usecols=COLS,low_memory=False)
 d=d[d.season.eq(season)&d.season_type.eq('REG')].copy()
 for c in ['posteam','defteam','home_team','away_team']:d[c]=d[c].replace({'LA':'LAR'})
 final=d.sort_values(['game_id','play_id']).groupby('game_id').tail(1)
 games=final[final.desc.fillna('').str.contains('END GAME',case=False)].copy()
 if games.empty:raise ValueError(f'No completed regular season games for {season}')
 q=d[d.game_id.isin(games.game_id)&d.play_type.isin(['pass','run'])&d.epa.notna()&d.posteam.notna()&d.defteam.notna()&d.qb_kneel.ne(1)&d.qb_spike.ne(1)].copy()
 q['ok']=q.epa.gt(0).astype(float)
 q['explosive']=((q.play_type.eq('pass')&q.yards_gained.ge(20))|(q.play_type.eq('run')&q.yards_gained.ge(10))).astype(float)
 q['dropback']=q.qb_dropback.eq(1);q['designed']=q.rush.eq(1)&~q.dropback
 return games,q

def aggregate(games,plays,recent=False):
 rows=[]
 for team in TEAMS:
  gs=games[games.home_team.eq(team)|games.away_team.eq(team)].sort_values(['game_date','game_id'])
  if recent:gs=gs.tail(4)
  if gs.empty:raise ValueError('Missing completed game data for '+team)
  q=plays[plays.game_id.isin(gs.game_id)];o=q[q.posteam.eq(team)];a=q[q.defteam.eq(team)]
  values=[]
  for z in [o,a]:values.extend([z.epa.mean(),z.loc[z.dropback,'epa'].mean(),z.loc[z.designed,'epa'].mean(),z.ok.mean(),z.explosive.mean(),z.loc[z.dropback,'sack'].mean()])
  pf=float(np.where(gs.home_team.eq(team),gs.home_score,gs.away_score).mean());pa=float(np.where(gs.home_team.eq(team),gs.away_score,gs.home_score).mean())
  nums=[float(v) for v in values]+[pf,pa]
  if not np.isfinite(nums).all():raise ValueError('Missing metrics for '+team)
  for ix in [3,4,5,9,10,11]:
   if not 0<=nums[ix]<=1:raise ValueError('Invalid rate')
  rows.append([team,len(gs)]+nums+[str(gs.game_date.max()),'nflverse completed REG games; pass/run; no kneels/spikes/no_play'])
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--season',type=int,default=2026);ap.add_argument('--current-file');ap.add_argument('--prior-file');ap.add_argument('--skip-prior',action='store_true');ap.add_argument('--output',default='data/model_data.json');args=ap.parse_args()
 games,plays=load(args.season,args.current_file)
 tables={'Team Stats':aggregate(games,plays),'Recent Stats':aggregate(games,plays,True)}
 if not args.skip_prior:
  old,oldplays=load(args.season-1,args.prior_file);tables['Prior Stats']=aggregate(old,oldplays)
 payload={'schema_version':1,'season':args.season,'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_through':str(games.game_date.max()),'completed_games':len(games),'tables':tables}
 out=pathlib.Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(payload,allow_nan=False,separators=(',',':')));os.replace(tmp,out)
 print(f'Validated {len(games)} completed games through {payload["source_through"]}; {len(tables)} tables')
if __name__=='__main__':main()

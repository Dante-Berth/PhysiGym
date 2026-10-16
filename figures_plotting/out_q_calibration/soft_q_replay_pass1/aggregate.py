import json,glob,sys,numpy as np,pandas as pd
rows=[]
for host,d in (("sureli9","rect2net"),("sureli11","net2rect")):
    for f in glob.glob(f"/tmp/claude-12501/-home-disc-a-bertin-Documents-Git-phd/40614011-a14f-4bd9-8bc8-7cfe40147aaf/scratchpad/softq/{host}/uniform_*.json"):
        tag=f.split("/")[-1][:-5]; _,m,s=tag.split("_"); 
        for r in json.load(open(f)): r.update(direction=d,mode=m,seed=int(s[4:])); rows.append(r)
df=pd.DataFrame(rows); df.to_csv("/tmp/claude-12501/-home-disc-a-bertin-Documents-Git-phd/40614011-a14f-4bd9-8bc8-7cfe40147aaf/scratchpad/softq/all_episodes.csv",index=False)
cols=["b_logged","b_dec_plain","b_dec_soft","ent_term","mean_logp","alpha","m_logged","m_dec_soft"]
per_seed=df.groupby(["direction","mode","seed"])[cols].mean().reset_index()
g=per_seed.groupby(["direction","mode"])
out=g[cols].mean().round(2); out["n_seeds"]=g.size(); out["n_ep"]=df.groupby(["direction","mode"]).size()
print(out.to_string())
print("\nsd across seeds"); print(g[["b_logged","b_dec_soft"]].std().round(1).to_string())

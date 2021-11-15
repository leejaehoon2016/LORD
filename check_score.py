import json, os
import pandas as pd
pd.set_option('display.max_rows', None)
def read_json(path):
    with open(path) as f:
        file = json.load(f)
    return file

lord = "lord/result/score"
base = "baselines/result/score"

df_lst = []
for i in os.listdir(lord):
    try:
        file = read_json(f"{lord}/{i}/result.json")['loss'][1]
        file = {k:round(v,3) for k,v in file.items()}
        tmp = i.split("_")
        
        file["Data"] = tmp[1]
        file['Model'] = 'lord'
        file["D(1)"] = tmp[2]
        file["D2"] = tmp[3]
        file["P"] = tmp[4]
        df_lst.append(pd.Series(file))
    except:
        pass
for i in os.listdir(base):
    try:
        file = read_json(f"{base}/{i}/result.json")['loss'][1]
        file = {k:round(v,3) for k,v in file.items()}
        tmp = i.split("_")
        if "nrde" in tmp[0]:
            file['Model'] = tmp[0]
            file["Data"] = tmp[3]
            file["D(1)"] = tmp[1]
            file["P"] = tmp[4]
        else:
            file['Model'] = tmp[0]
            file["Data"] = tmp[2]
            file["D(1)"] = ""
            file["P"] = tmp[3]
        file["D2"] = ""
        df_lst.append(pd.Series(file))
    except:
        pass
df = pd.concat(df_lst,axis=1).T
df = df[['Data', 'Model', 'D(1)', 'D2', 'P', "r2", "explained_variance","mean_squared","mean_absolute"]]
# df[["r2", "explained_variance","mean_squared","mean_absolute"]] = df[["r2", "explained_variance","mean_squared","mean_absolute"]].round(0)
df["P"] = df["P"].astype(int)
df = df.sort_values(["Data", "Model", "D(1)", "D2", "P"]).reset_index(drop=True)
print(df)



    


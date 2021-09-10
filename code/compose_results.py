import os
import pandas as pd
import json
import csv

csv_file = "result.csv"
exps = os.listdir("../experiments")
count = 0
res = []
for exp in exps:
    exp = os.path.join("../experiments", exp)
    exp = os.path.join(exp, os.listdir(exp)[0], 'result')
    listdir = os.listdir(exp)
    listdir.remove("augmentedroc.pdf")
    filename = os.path.join(exp, listdir[0])
    try:
        with open(filename) as f:
            result = json.load(f)
            f.close()
        res.append(result)
    except Exception as ex:
        print("EXCEPTION: ", filename)
df = pd.DataFrame(res)
df.to_csv("result.csv")


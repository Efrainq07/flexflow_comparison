import argparse
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--input','-i',default="FF_profiling.txt")
parser.add_argument('--output','-o',default="FF_profiling.csv")
args = parser.parse_args()
file = open(args.input,"r")
lines = file.readlines()
linef = []
for line in lines:
    linef.append(line.split("size:")[-1].replace("\n","").split("\t"))
nparr = np.array(linef)
df = pd.DataFrame(data=nparr,columns=["size","name"])
df["size"]=df["size"].astype(float)/2**20
df.to_csv(args.output,index=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
file.close()

import pandas as pd

train = pd.read_csv('submission_0504.csv', encoding="ISO-8859-1")#[:1000]

train['Active_Customer'] = train['Active_Customer'].map(lambda x: round(x))

train.to_csv("modified_0504.csv", index=False)

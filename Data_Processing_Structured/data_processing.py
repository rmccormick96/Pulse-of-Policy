
### Federal Election Comission data
### Prepare data from downloaded Excel file

import pandas as pd

# Candidates 2017-2018. All candidates
# https://www.fec.gov/campaign-finance-data/all-candidates-file-description/
# https://www.fec.gov/data/browse-data/?tab=bulk-data

cols = ['CAND_ID','CAND_NAME','CAND_ICI','PTY_CD','CAND_PTY_AFFILIATION',
        'TTL_RECEIPTS','TRANS_FROM_AUTH','TTL_DISB','TRANS_TO_AUTH','COH_BOP',
        'COH_COP','CAND_CONTRIB','CAND_LOANS','OTHER_LOANS','CAND_LOAN_REPAY',
        'OTHER_LOAN_REPAY','DEBTS_OWED_BY','TTL_INDIV_CONTRIB','CAND_OFFICE_ST',
        'CAND_OFFICE_DISTRICT','SPEC_ELECTION','PRIM_ELECTION','RUN_ELECTION',
        'GEN_ELECTION','GEN_ELECTION_PRECENT','OTHER_POL_CMTE_CONTRIB',
        'POL_PTY_CONTRIB','CVG_END_DT','INDIV_REFUNDS','CMTE_REFUNDS']


df_com = pd.read_csv('Data/fed_election_comission/weball18.txt', 
                 sep = '|', header = None, names = cols)
df_com.loc[:, "CAND_PTY_AFFILIATION"].value_counts()

cols_excel = ['CAND_ID', 'CAND_NAME', 'CAND_PTY_AFFILIATION', 'CVG_END_DT', 
              'TTL_RECEIPTS', 'TRANS_FROM_AUTH', 'TTL_DISB', 'TRANS_TO_AUTH', 
              'CAND_LOANS', 'DEBTS_OWED_BY', 'TTL_INDIV_CONTRIB']
df_com_excel = df_com.loc[:, cols_excel]
# CVG_END_DT is coverage end date

# Party code
df_com.value_counts('PTY_CD')

# Party affiliation
df_com.value_counts('CAND_PTY_AFFILIATION')

# Total receipts
df_com.loc[:, 'TTL_RECEIPTS']
df_com.loc[:, 'TTL_RECEIPTS'].describe()

## Group by Party Code: PTY_CD
cols_keep = ['PTY_CD', 'TTL_RECEIPTS', 'TRANS_FROM_AUTH', 'TTL_DISB', 
             'TRANS_TO_AUTH', 'CAND_CONTRIB', 'CAND_LOANS', 'OTHER_LOANS', 
             'CAND_LOAN_REPAY', 'OTHER_LOAN_REPAY', 'DEBTS_OWED_BY', 
             'TTL_INDIV_CONTRIB']
df_g = df_com.groupby('PTY_CD', as_index = False).sum().loc[:, cols_keep]


### Bill information from Pro Publica
### 

import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

path_bills = "Data/bills/115th_congress_house_bills_base.json"

with open(path_bills, "r", encoding = 'utf-8') as file:
    bills = json.load(file)

df_bills = pd.DataFrame(bills)
df_bills.loc[:, 'bill_type'].value_counts()

df_bills = df_bills.loc[(df_bills.loc[:, "bill_type"] == "hr") | (df_bills.loc[:, "bill_type"] == "hjres")]

# Passage
ratio_passed = 1 - (df_bills.loc[:, 'house_passage'].isnull().sum() / df_bills.shape[0])
ratio_passed

#
passage = df_bills.loc[:, 'house_passage'].dropna()

### Histogram of bills that did pass
fig, ax = plt.subplots(figsize = (20, 10))
plt.hist(passage, bins = 10, 
         alpha = 0.9, 
         edgecolor='black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_major_locator(MaxNLocator(nbins = 10))

plt.tick_params(axis='both',       # Apply changes to both x and y axes
                which='major',     # Apply changes to major ticks
                labelsize = 15)         # Increase tick length
plt.gca().invert_xaxis()
plt.show()

## 
# list_com = [q.split(", ")[1] + q.split(", ")[0] for q in list(df_com.loc[:, "CAND_NAME"])]

list_com = [q.split(", ") for q in list(df_com.loc[:, "CAND_NAME"])]

list_new = []
for q in list_com:
    if len(q) == 2:
        newstring = q[1] + " " + q[0]
        list_new.append(newstring)
    else:
        list_new.append(q[0])
        
df_com.loc[:, "sponsor_name"] = list_new        
igit = df_com.loc[:, ["CAND_NAME", "sponsor_name"]]

### Merge Federal Election Comission data with Propublica Data
from fuzzywuzzy import fuzz

# set_1 = set(df_bills.loc[:, 'sponsor_name'].str.lower())
# set_2 = set(df_com.loc[:, 'sponsor_name'].str.lower())

df_bills.loc[:, 'sponsor_name'] = df_bills.loc[:, 'sponsor_name'].str.lower()
df_com.loc[:, 'sponsor_name'] = df_com.loc[:, 'sponsor_name'].str.lower()

set_1 = set(df_bills.loc[:, 'sponsor_name'])
set_2 = set(df_com.loc[:, 'sponsor_name'])

list_matches = []
i = 0
for name_1 in set_1:
    best_match = 0
    for name_2 in set_2:
        ratio = fuzz.ratio(name_1, name_2)
        if ratio > best_match:
            best_match = ratio
            name_2_best = name_2
    if best_match < 80:
        continue
    else:
        list_matches.append([name_1, name_2_best])
        
    i += 1
    if i // 10 == i / 10:
        print(i)

df_bills.loc[:, 'sponsor_name_merge'] = 9999
df_com.loc[:, 'sponsor_name_merge'] = 9999

for i, (name_1, name_2) in enumerate(list_matches):
    df_bills.loc[df_bills.loc[:, 'sponsor_name'] == name_1, 'sponsor_name_merge'] = i
    df_com.loc[df_com.loc[:, 'sponsor_name'] == name_2, 'sponsor_name_merge'] = i
        
df_com = df_com.loc[df_com.loc[:, 'sponsor_name_merge'] != 9999]    

df_merged = df_bills.merge(df_com, how = "left", on = "sponsor_name_merge")
    
# # 
# df_merged.loc[:, 'bill_type'].value_counts()
# df_merged.loc[:, 'sponsor_party'].value_counts()

# Create bipartisanship ratio
list_cosponsors = list(df_merged.loc[:, 'cosponsors_by_party'])
list_rep = []
list_dem = []

for cosp in list_cosponsors:
    dem = cosp.get('D', 0)
    rep = cosp.get('R', 0)
    list_rep.append(rep)
    list_dem.append(dem)

df_merged.loc[:, 'cosp_dem'] = list_dem
df_merged.loc[:, 'cosp_rep'] = list_rep

df_merged.loc[:, 'cosp_ratio'] = df_merged.loc[:, 'cosp_dem'] / (df_merged.loc[:, 'cosp_rep'] + 1)

df_merged.loc[:, 'spons_dem'] = 0
df_merged.loc[df_merged.loc[:, 'sponsor_party'] == 'D', 'spons_dem'] = 1

cols_feat = ['bill_id', 'bill_slug', 'number', 'sponsor_party', 'introduced_date', 'house_passage',
             'TTL_RECEIPTS',  'TRANS_FROM_AUTH', 
             'TTL_DISB', 'TRANS_TO_AUTH', 'CAND_CONTRIB', 'CAND_LOANS', 
             'OTHER_LOANS', 'CAND_LOAN_REPAY', 'OTHER_LOAN_REPAY', 
             'DEBTS_OWED_BY', 'TTL_INDIV_CONTRIB', 'OTHER_POL_CMTE_CONTRIB',
             'POL_PTY_CONTRIB', 'CVG_END_DT', 'INDIV_REFUNDS', 'CMTE_REFUNDS',
             'cosp_dem', 'cosp_rep', 'cosp_ratio', 'spons_dem']


df_merged = df_merged.loc[:, cols_feat]

type(df_merged.loc[:, 'introduced_date'].iloc[0])

df_merged.to_csv('Data/df_merged.txt', index = False)


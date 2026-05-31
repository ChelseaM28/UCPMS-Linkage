#Fixing the department names in the priority file after user error
import pandas as pd

priority = pd.read_csv('/workspaces/Scholarly-Impact---Automation-Pipeline-Work/src/data/priority_data.csv')
original = pd.read_csv('/workspaces/Scholarly-Impact---Automation-Pipeline-Work/src/data/original.csv')  # the correct one

# lookup dict: scopus_id -> department from original
original_scopus = original[original['Identifier scheme'] == 'Scopus ID']
dept_lookup = dict(zip(original_scopus['Identifier'].astype(str), 
                       original_scopus['Department']))

# Replace department in priority where scopus id matches
priority['Identifier'] = priority['Identifier'].astype(str)
priority['Department'] = priority.apply(
    lambda row: dept_lookup.get(row['Identifier'], row['Department']) 
    if row['Identifier scheme'] == 'Scopus ID' else row['Department'],
    axis=1
)

priority.to_csv('priority_data_updated.csv', index=False)

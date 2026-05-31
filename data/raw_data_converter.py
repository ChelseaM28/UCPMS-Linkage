#this script will take imported data and convert it into a format that can be used for the features. 
# Author: Chelsea Momoh
# Date: 2026-04-26
# Version: 1.0
#Filename: raw_data_converter



import json
import pandas as pd

with open('/workspaces/Scholarly-Impact---Automation-Pipeline-Work/src/data/scopus_cleaned_merged.json', 'r') as f:
    scopus_cleaned_merged = json.load(f)

list_of_scopus_ids_from_api = list(scopus_cleaned_merged.keys())

################## RETRIEVE UCPMS DATA ##########################
#@Brief: Returns a dict keyed by Scopus ID so I can match by ID, not index.
#        Fields are joined into a single string to match Scopus format.

def retrieve_UCPMS_data(list_of_scopus_ids_from_api):
    print("Running retrieve_ucpms_data")
    df = pd.read_csv('/workspaces/Scholarly-Impact---Automation-Pipeline-Work/src/data/priority_data_updated.csv')
    df = df[df.index < 2779] #MADE A CHANGE HERE
    df = df[df['Identifier scheme'] == 'Scopus ID']
    
    df = df[df['Identifier'].isin(list_of_scopus_ids_from_api)]
    df = df[df['Name'].notna()]  

  # Key by Scopus ID so zipper can match by ID, not position
    ucpms_by_id = {}
    for _, row in df.iterrows():
        scopus_id = str(row['Identifier'])
        ucpms_by_id[scopus_id] = {
            "name": row['Name'],
            "fields": str(row['Department']) if pd.notna(row['Department']) else ""
            # No institution column in UCPMS, field doubles as context
        }
   
    print("Completed running retrieve_UCPMS_data")
    return ucpms_by_id


################## LISTIFYING SCOPUS DATA ##########################
#@Brief: Converts Scopus fields (list) and affiliations (list of dicts) into
#        single strings per author. Returns a dict keyed by Scopus ID.

def listifying_scopus_data(scopus_api_data, list_of_scopus_ids_from_api):
    print("Running listifying_scopus_data")
    scopus_by_id = {}

    for scopus_id, data in scopus_api_data.items():
        if scopus_id not in list_of_scopus_ids_from_api:
            continue
        if not data.get("name"): 
          continue
        # Fields: join list into one string
        fields_str = ", ".join(data.get("fields", []))

        # Affiliations: flatten each dict into a string, then join all
        aff_parts = []
        for aff in data.get("affiliations", []):
            aff_parts.append(
                f"{aff['institution']} ({aff['city']}, {aff['country']})"
            )
        
        affiliations_str = "; ".join(aff_parts)

        scopus_by_id[scopus_id] = {
            "name": data["name"],
            "fields": fields_str,
            "affiliations": affiliations_str
        }
    print("Completed running listifying_scopus_data")
    return scopus_by_id


################## ZIPPER ##########################
#@Brief: Matches UCPMS and Scopus entries by Scopus ID.
#        Produces three parallel lists of (scopus_value, ucpms_value) pairs.

def zipper(ucpms_by_id, scopus_by_id):
    print("Running Zipper")
    name_pairs = []
    field_pairs = []
    institution_pairs = []

    # Only zip IDs present in BOTH sources
    shared_ids = sorted(set(ucpms_by_id.keys()) & set(scopus_by_id.keys()))

    for scopus_id in shared_ids:
        ucpms = ucpms_by_id[scopus_id]
        scopus = scopus_by_id[scopus_id]

        #Additional redundancy for None values
        if not scopus["name"] or not ucpms["name"]:  
          #print(f"Skipping {scopus_id} — missing name on one side")
          continue

        name_pairs.append((scopus["name"], ucpms["name"]))
        field_pairs.append((scopus["fields"], ucpms["fields"]))
        # UCPMS has no institution column, use empty string as placeholder
        institution_pairs.append((scopus["affiliations"], ""))

    print("Completed running Zipper")
    return name_pairs, field_pairs, institution_pairs, shared_ids


################## MAIN ##########################
#@Brief: Creates the name pairs, field pairs, and institution pairs, as well as 
#        returning a list of the SCOPUS IDs that are being verified.
def main():
  print("Running raw data converter.main")
  ucpms_by_id = retrieve_UCPMS_data(list_of_scopus_ids_from_api)
  scopus_by_id = listifying_scopus_data(scopus_cleaned_merged, list_of_scopus_ids_from_api)

  name_pairs, field_pairs, institution_pairs, shared_ids = zipper(ucpms_by_id, scopus_by_id)
  print("Completed running raw_data_converter.main")
  return name_pairs, field_pairs, institution_pairs, shared_ids


import os
import pybliometrics
from pathlib import Path
import pandas as pd
api_key = os.getenv('API_TOKEN')
inst_token = os.getenv('INSTTOKEN')
pybliometrics.init(keys = [api_key] , inst_tokens = [inst_token])

from pybliometrics.scopus import AuthorRetrieval, AffiliationRetrieval, SubjectClassifications,AbstractRetrieval 



def retrieve_scopus_data():
    print("Running retrieve_scopus_data...")
    scopus_api_data = {}
    df = pd.read_csv('src/data/original.csv') #this is the csv file that contains the scopus ids and other data that I will use for training the model

    df = df[7929:] #This is how I continued to retrive after my limit ran out in week 1
    df = df[df['Identifier scheme'] == 'Scopus ID']
    df = df[df['Identifier'] != 'NaN']
    
    failed_ids = []

    for scopus_id in df['Identifier']: #this is the column in the csv file that contains the scopus ids
        try:
            author = AuthorRetrieval(author_id=scopus_id, view = 'ENHANCED')
            affiliation = author.affiliation_current
            field = author.subject_areas
    
            scopus_api_data[scopus_id] = {"author name": author, "affiliation": affiliation, "field": field}
        except Exception as e:
            print(f"  ✗ Failed for ID {scopus_id}: {e}")
            failed_ids.append(scopus_id)
    print("Completed running retrieve_scopus_data.  Data has been saved.\n")
    return scopus_api_data

def clean_author_data(scopus_api_data):
    cleaned = {}
    
    for scopus_id, data in scopus_api_data.items():
        author = data["author name"]
        
        # Extract actual name
        name = f"{author.given_name} {author.surname}" if author.given_name and author.surname else author.indexed_name

        # Extract affiliations -> list of institution names
        affiliations = []
        if data["affiliation"]:
            for aff in data["affiliation"]:
                affiliations.append({
                    "institution": aff.preferred_name,
                    "city": aff.city,
                    "country": aff.country
                })

        # Extract subject areas -> list of area names only
        fields = []
        if data["field"]:
            for f in data["field"]:
                fields.append(f.area)

        cleaned[scopus_id] = {
            "name": name,
            "affiliations": affiliations,
            #"co_authors": co_authors,
            "fields": fields
        }

    return cleaned

selected_data = clean_author_data(retrieve_scopus_data())
import json
with open('scopus_cleaned_v3.json', 'w') as f:
    json.dump(selected_data, f, indent=2)

'''
#Code was used to merge json files
#Now I need to append the next json file:
# Load both files
with open('scopus_cleaned_v2.json', 'r') as f:  #  original file
    data_v1 = json.load(f)

with open('scopus_cleaned_v3.json', 'r') as f:
    data_v3 = json.load(f)

merged = data_v1.copy()
merged.update(data_v3)

# Save the merged result
with open('scopus_cleaned_merged.json', 'w') as f:
    json.dump(merged, f, indent=2)

print(f"v1 records: {len(data_v1)}")
print(f"v3 records: {len(data_v3)}")
print(f"Merged records: {len(merged)}")
'''
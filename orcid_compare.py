#This script will implement the orcid compare feature
# Author: Chelsea Momoh
# Date: 2026-05-13
# Version: 1.0
#Filename: orcid_compare.py

'''
Firstly, I'm going to wrangle the csv file to find out how many SCOPUS IDs actually have a 
corresponding ORCID to see if the feature will be worthwhile.

If so, I will need to create a function that generates a list of CLAIMED (Yes in the 
"Claimed" column) ORCID IDs and then use the API to generate a list of SCOPUS IDs that have 
links to the ORCID ID. 

This feature will add a point to the overall similarity score of the SCOPUS IDs for which 
a match could be made. Otherwise, the feature will have a similarity score of 0 for 
"no indicator."
'''

import pandas as pd
data = pd.read_csv("/workspaces/Scholarly-Impact---Automation-Pipeline-Work/src/data/original.csv")

'''
def orcid_csv_wrangle(data):
1. group by name
2  if the group has an orcid ID, keep
3. If the group's orcid id has an id that has been claimed, keep
4. print the number of groups left.

IF the number seems valuable:
1. From the groups, create a list of claimed ORCID IDs
#2 and 3 MIGHT NEED TO COME FROM THE API.PY SCRIPT
2. have an api pull the SCOPUS IDs associated with the claimed orcid ids
3. return the associated_orcid_SCOPUS_IDs
4. determine a method to ensure that for every field/name/area pair list, there is
   a list of their SCOPUS ID in the EXACT SAME ORDER. (i might already have this.)
   This list of scopus IDs might currently be called shared_ids.
'''

'''
#THIS FUNCTION NEEDS TO GO IN THE FEATURE FOLDER
def orcid_compare(possibly shared_ids):
1.  Recall that features iterate through each name in the list to assign 
    a feature score. If the SCOPUS ID matches one of the SCOPUS IDs saved in the
    associated_orcid_SCOPUS_IDs, add a point for this feature. 

that's it! return the list of points/non points and it will be zipped into the 
feature set for the model.
'''
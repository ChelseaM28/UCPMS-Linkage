"""
filename: disambiiguation_feature
Name Disambiguation Features.
As per the automation plan, this script will complete steps 3-4 of the ML 
pipeline. 
"""

# Author: Chelsea Momoh
# Date: 2026-04-06
# Version: 1.0


'''data = [
    ("A. Ennamaria B", "AMENTA, Ennamaria B"),
    ("AMES, James B", "Ames, James"),
    ("ANDERSON, D", "ANDERSON, Damien"),
    ("ARMIEN, Anibal G", "Armién, Aníbal Guillermo"),
    ("JOHNSTON, Warren E", "Johnston, W. R.")
]

data_test = [
    ("B. Carlos M", "MENDEZ, Carlos B"),       
    ("SMITH, John A", "Smuth, John"),         
    ("LEE, D", "LEE, Daniel"),              
    ("NGUYEN, Linh T", "Nguyen, Linh Thi"),      
    ("GARCIA, Ramon F", "Garcia, R. J.") 
]'''

##################NAME UNSCRAMBLING DETECTION##########################
#@Brief: Checking the order of commas to determine the order of the name. 
#        UCMPS and SCOPUS formatting frequently differs in this way.
#        e.g: AMENTA, Ennamaria B = Ennamaria B Amenta, while
#             A. Ennamaria B = A. Ennamaria B. 

def normalize_format(name):
    if name is None:
        return []
    tokens = name.replace(",", " ").split()
    if len(tokens) >= 2 and len(tokens[0]) == 1:
        tokens = tokens[1:] + [tokens[0]]
    return tokens

##################INITIALS DETECTION##########################
#@brief: This function catches scenarios where one name is a single initial and 
#        the other is a full name. It will return "inconclusive" if both are 
#        initials (since we can't determine if they match or not).
#        It should ultimately be weighed more strongly than the name similarity 
#        feature; if the only information we have is an initial, we can 
#        only check for consistency between the initial and the full name, which 
#        is a stronger signal than basic text similarity.


def compare_token(t1, t2):
    #If either token is a single initial, we can only check initial consistency
    if len(t1) == 1 or len(t2) == 1:
        if t1[0] != t2[0]:
            return "reject"
        return "inconclusive"
    #Both are full names => require exact match
    if t1 == t2:
        return "accept"
    return "reject"


##################NAME SIMILARITY FEATURE##########################
#@brief: Adding a feature based on basic text similarity from difflib library.
#        "The ratio is computed as 2.0 × M / T, where M is the total number of 
#        matching characters across all identified subsequences, and T is the 
#        total number of characters in both strings combined."

from difflib import SequenceMatcher
def name_similarity(data):
    print("Running name_similarity")
    similarities = []
    for n1, n2 in data:
        if n1 is None or n2 is None:
            similarities.append(0.0)
            continue
        n1_list = sorted([token.lower() for token in normalize_format(n1)])  # 👈
        n2_list = sorted([token.lower() for token in normalize_format(n2)]) 
        n1_list = [token.lower() for token in normalize_format(n1)]
        n2_list = [token.lower() for token in normalize_format(n2)]
        similarity = SequenceMatcher(None, n1_list, n2_list).ratio()
        similarities.append(similarity)
    
    print("Completed running name_similarity")
    return similarities


##################NAME DISAMBIGUATION##########################
#@Brief: This function will incorporate the name unscrambling and initals detection features.
#        It removes special characters that may confuse the model.
#        It simulates my initial "glance" test for name matching. It will return a list 
#        of matches (1 for match, 0 for no match, 2 for inconclusive) and a log of notable catches.

def disambiguate_names(data):
    print("Running: 'disambiguate_names'...")
    special_characters = {"í": "i", "é": "e", "á": "a", "ó": "o", "ú": "u"}
    notable_catches = {} #I might decide to use this for logging notable catches (inconclusive cases) that the model can learn from.
    matches = [] #This contains the results of the disambiguation (1 for match, 0 for no match, 2 for inconclusive)
    
    #Replacing all the special characters for normalization
    for n1, n2 in data:
        for special, replace in special_characters.items():
            n1 = n1.replace(special, replace)
            n2 = n2.replace(special, replace)
            
        n1 = n1.replace(".", "").lower()
        n2 = n2.replace(".", "").lower()

        n1_split = sorted(normalize_format(n1))
        n2_split = sorted(normalize_format(n2))
        #print(f"Comparing '{n1_split}' and '{n2_split}'...\n")

        #Looking for the middle initial. It will be the last token if there are 3 or more tokens in a name
        if len(n1_split) >= 3 and len(n2_split) >= 3:
            if n1_split[-1][:1] != n2_split[-1][:1]:
                #print(f"{n1} and {n2} do not match due to different middle initials.\n")
                matches.append(0)
                continue
        
       
        last_result = compare_token(n1_split[0], n2_split[0])
        first_result = compare_token(n1_split[1], n2_split[1])


        if last_result == "reject" or first_result == "reject":
            #print(f"{n1} and {n2} do not match.\n")
            matches.append(0)
        elif last_result == "accept" and first_result == "accept":
            #print(f"{n1} and {n2} match.\n")
            matches.append(1)
        else: #This will run if there are only initials available, which is not enough information to 
            #make a determination.
            #print(f"{n1} and {n2} are inconclusive.\n")
            matches.append(2)
            notable_catches[(n1, n2)] = "inconclusive"
    print("Completed running: 'disambiguate_names")
    return matches



#REVISIT THIS!!: 
#I don't know if the ML model can output 2 for inconclusive cases, but I will include it in the training data and see 
# if the model can learn to identify those cases as well.
'''
model = LogisticRegression()
model.fit(X_train, y_train)

print(f"The machine learning model predicts: {model.predict(X_test)}")
#print("bingo")'''

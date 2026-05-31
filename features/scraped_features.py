"""
filename: scraped_features
Name Disambiguation Features.
As per the automation plan, this script will complete steps 3-4 of the ML 
pipeline. 
"""

# Author: Chelsea Momoh
# Date: 2026-04-06
# Version: 1.0



##################RESEARCH AREA/PUBLICATION TITLE SIMILARITY FEATURE##########################
#@Brief: This function checks for MEANING similarity between field of research/subject area keywords.
#        It will check the UCPMS field against the SCOPUS field, and return a numical score 
#        based on the accuracy of shared keywords.
#        It will also check for similarity across UCPMS publication titles and SCOPUS publication titles.


#wup_similarity from NLTK WordNet

from rapidfuzz import fuzz, process, utils
from nltk.corpus import wordnet as wn
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('wordnet_ic')
from nltk.corpus import wordnet_ic
nltk.data.find('corpora/wordnet_ic') 
brown_ic = wordnet_ic.ic('ic-brown.dat')
from itertools import product

def token_overlap(a, b):
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    return len(a_set & b_set) / max(len(a_set), 1)

def domain_penalty(a, b):
    domains = [
        "psychology", "plant", "medicine", "chemistry",
        "biology", "pathology", "viticulture", "enology"
    ]

    a, b = a.lower(), b.lower()

    for d1 in domains:
        for d2 in domains:
            if d1 in a and d2 in b and d1 != d2:
                return 0.5
    return 0

# ---------------- helper: best synset match ----------------
def best_pair(s1_list, s2_list):
    best = 0
    for s1 in s1_list:
        for s2 in s2_list:
            try:
                score = s1.wup_similarity(s2) or 0
                best = max(best, score)
            except:
                continue
    return best

#NLTK implementation
def research_area_similarity(data):
    print("Running: 'research_area_similarity'...")

    wun_similarity_scores = []
    resnik_similarity_scores = []
    lin_similarity_scores = []
    nltk_scores = []

    for ucmps_field, scopus_field in data:
        # Clean the scopus field before splitting
        scopus_field = scopus_field.replace('MED:', '').replace('|', ' ').strip()
        ucmps_field = ucmps_field.replace('(all),', '').replace('(miscellaneous),', '').strip()
        ucmps_field = ucmps_field.split()[0:2]
        scopus_field = scopus_field.split()[0:2]

        if not ucmps_field or not scopus_field:
            nltk_scores.append(0.0)
            wun_similarity_scores.append([0, 0, 0, 0])
            resnik_similarity_scores.append([0, 0, 0, 0])
            lin_similarity_scores.append([0, 0, 0, 0])
            continue

        ucmps_synsets_first = wn.synsets(ucmps_field[0])
        scopus_synsets_first = wn.synsets(scopus_field[0])

        ucmps_synsets_second = wn.synsets(ucmps_field[1]) if len(ucmps_field) > 1 else []
        scopus_synsets_second = wn.synsets(scopus_field[1]) if len(scopus_field) > 1 else []

        

        # ---------------- FIRST / FIRST ----------------
        try:
            similarity_first_to_first = best_pair(ucmps_synsets_first, scopus_synsets_first)
            resnik_first_to_first = wn.res_similarity(
                ucmps_synsets_first[0],
                scopus_synsets_first[0],
                ic=brown_ic
            )
            lin_first_to_first = wn.lin_similarity(
                ucmps_synsets_first[0],
                scopus_synsets_first[0],
                ic=brown_ic
            )
        except:
            similarity_first_to_first = 0
            resnik_first_to_first = 0
            lin_first_to_first = 0

        # ---------------- FIRST / SECOND ----------------
        try:
            if scopus_synsets_second and "science" not in scopus_synsets_second[0].name().lower():
                similarity_first_to_second = best_pair(ucmps_synsets_first, scopus_synsets_second)
                resnik_first_to_second = wn.res_similarity(
                    ucmps_synsets_first[0],
                    scopus_synsets_second[0],
                    ic=brown_ic
                )
                lin_first_to_second = wn.lin_similarity(
                    ucmps_synsets_first[0],
                    scopus_synsets_second[0],
                    ic=brown_ic
                )
            else:
                similarity_first_to_second = -1
                resnik_first_to_second = -1
                lin_first_to_second = -1
        except:
            similarity_first_to_second = 0
            resnik_first_to_second = 0
            lin_first_to_second = 0

        # ---------------- SECOND / FIRST ----------------
        try:
            if ucmps_synsets_second and "science" not in ucmps_synsets_second[0].name().lower():
                similarity_second_to_first = best_pair(ucmps_synsets_second, scopus_synsets_first)
                resnik_second_to_first = wn.res_similarity(
                    ucmps_synsets_second[0],
                    scopus_synsets_first[0],
                    ic=brown_ic
                )
                lin_second_to_first = wn.lin_similarity(
                    ucmps_synsets_second[0],
                    scopus_synsets_first[0],
                    ic=brown_ic
                )
            else:
                similarity_second_to_first = -1
                resnik_second_to_first = -1
                lin_second_to_first = -1
        except:
            similarity_second_to_first = 0
            resnik_second_to_first = 0
            lin_second_to_first = 0

        # ---------------- SECOND / SECOND ----------------
        try:
            if (
                ucmps_synsets_second
                and scopus_synsets_second
                and "science" not in ucmps_synsets_second[0].name().lower()
                and "science" not in scopus_synsets_second[0].name().lower()
            ):
                similarity_second_to_second = best_pair(ucmps_synsets_second, scopus_synsets_second)
                resnik_second_to_second = wn.res_similarity(
                    ucmps_synsets_second[0],
                    scopus_synsets_second[0],
                    ic=brown_ic
                )
                lin_second_to_second = wn.lin_similarity(
                    ucmps_synsets_second[0],
                    scopus_synsets_second[0],
                    ic=brown_ic
                )
            else:
                similarity_second_to_second = -1
                resnik_second_to_second = -1
                lin_second_to_second = -1
        except:
            similarity_second_to_second = 0
            resnik_second_to_second = 0
            lin_second_to_second = 0

        # ---------------- RAW ARRAYS ----------------
        wun_similarity_scores.append([
            similarity_first_to_first,
            similarity_first_to_second,
            similarity_second_to_first,
            similarity_second_to_second
        ])

        resnik_similarity_scores.append([
            resnik_first_to_first,
            resnik_first_to_second,
            resnik_second_to_first,
            resnik_second_to_second
        ])

        lin_similarity_scores.append([
            lin_first_to_first,
            lin_first_to_second,
            lin_second_to_first,
            lin_second_to_second
        ])

    # ---------------- ARRAYS ----------------
    wun_array = np.array(wun_similarity_scores)
    res_array = np.array(resnik_similarity_scores)
    lin_array = np.array(lin_similarity_scores)

    # ---------------- CLEAN + THRESHOLDING + DOMAIN FIX ----------------
    for arr in [wun_array, res_array, lin_array]:
        for row in range(arr.shape[0]):
            for column in range(arr.shape[1]):

                val = arr[row, column]

                if val == "error" or val is None:
                    arr[row, column] = 0
                    continue

                val = float(val)

                if val > 0.5:
                    val = 1
                elif val > 0.25:
                    val = 0.5
                else:
                    val = 0

                arr[row, column] = val

    # ---------------- SUM ----------------
    summed_wun_array = np.sum(wun_array.astype(float), axis=1)
    summed_res_array = np.sum(res_array.astype(float), axis=1)
    summed_lin_array = np.sum(lin_array.astype(float), axis=1)

    # ---------------- FINAL SCORE + IMPROVEMENTS ----------------
    for i in range(len(summed_wun_array)):

        base_score = (
            .5*summed_wun_array[i]
            + .25*summed_res_array[i]
            + .25*summed_lin_array[i]
        )

        
        field1 = data[i][0]
        field2 = data[i][1]

        penalty = domain_penalty(field1, field2)
        bonus = token_overlap(field1, field2)

        final_score = max(0, base_score - penalty + (bonus * 0.5))

        nltk_scores.append(float(final_score))
        

    
    print("Completed Running 'research_area_similarity'")
    return nltk_scores



#RapidFuzz Implementation
#This expanded dictionary helps rapid fuzz understand what the abbreviations mean
DEPT_EXPANSION = {
    # Engineering
    'ELECT': 'electrical',
    'COMP': 'computer',
    'ENGR': 'engineering',
    'MECH': 'mechanical',
    'AEROSPACE': 'aerospace',
    'CIVIL': 'civil',
    'CHEM': 'chemistry',
    'BIOCHEM': 'biochemistry',
    'BIOMEDICAL': 'biomedical',
    'MATERIALS': 'materials',
    
    # Medicine
    'MED': 'medicine',
    'INT': 'internal',
    'ANESTH': 'anesthesiology',
    'HMCTBMT': 'hematology bone marrow transplant',
    'HEM': 'hematology',
    'ONC': 'oncology',
    'SURG': 'surgery',
    'RAD': 'radiology',
    'ORTHO': 'orthopedic',
    'ORTHOPAEDIC': 'orthopedic',
    'OBSTETRICS': 'obstetrics',
    'GYNECOLOGY': 'gynecology',
    'NEPHROLOGY': 'nephrology',
    'NEUROLOGY': 'neurology',
    'NEUROLOGICAL': 'neurological',
    'OTOLARYNGOLOGY': 'otolaryngology',
    'PULMONARY': 'pulmonary',
    'GASTROENTEROLOGY': 'gastroenterology',
    'ENDOCRINOLOGY': 'endocrinology',
    'RHEUMATOLOGY': 'rheumatology',
    'DERMATOLOGY': 'dermatology',
    'PHARMACOLOGY': 'pharmacology',
    'PEDIATRICS': 'pediatrics',
    'PATHOLOGY': 'pathology',
    'MICROBIOLOGY': 'microbiology',
    'IMMUNOLOGY': 'immunology',
    'UROLOGIC': 'urology',
    'HOSPITALIST': 'hospital medicine',
    'GERIATRICS': 'geriatrics',
    'PSYCHIATRY': 'psychiatry',
    'RADIATION': 'radiation',
    'EMERGENCY': 'emergency',
    'REHAB': 'rehabilitation',
    'PHYSICAL': 'physical',
    
    # Veterinary
    'VM': 'veterinary medicine',
    'VET': 'veterinary',
    'ANAT': 'anatomy',
    'PHYSIO': 'physiology',
    'REPROD': 'reproduction',
    'HLTH': 'health',
    'CTR': 'center',
    
    # Sciences
    'BIO': 'biological',
    'AG': 'agricultural',
    'MOLEC': 'molecular',
    'MICRO': 'microbiology',
    'IMMUN': 'immunology',
    'GENETICS': 'genetics',
    'MOL': 'molecular',
    'EVOL': 'evolution',
    'ECOL': 'ecology',
    'ENV': 'environmental',
    'SCI': 'science',
    'NEURO': 'neuroscience',
    'NEMATOLOGY': 'nematology',
    'ENTOMOLOGY': 'entomology',
    'TOXICOLOGY': 'toxicology',
    'EPIDEMIOLOGY': 'epidemiology',
    
    # Social sciences / humanities
    'PSYCH': 'psychology',
    'ECON': 'economics',
    'POLI': 'political',
    'STDS': 'studies',
    'BEHAV': 'behavioral',
    'LING': 'linguistics',
    'LANG': 'language',
    'CULT': 'cultural',
    'SOCIOL': 'sociology',
    'ANTHRO': 'anthropology',
    'GENDERSEXUALITY': 'gender sexuality',
    'WOMENSSTUDIES': 'women studies',
    'CHICANO': 'chicano',
    'AFRICAN': 'african',
    'ASIAN': 'asian',
    'AMERICAN': 'american',
    'COMPARATIVE': 'comparative',
    'THEATRE': 'theatre',
    'CINEMA': 'cinema',
    'VITICULTURE': 'viticulture',
    'ENOLOGY': 'enology',
    
    # Misc
    'RESOURCE': 'resource',
    'RESOURCES': 'resources',
    'LAND': 'land',
    'AIR': 'air',
    'WATER': 'water',
    'FOOD': 'food',
    'NUTRITION': 'nutrition',
    'POPULATION': 'population',
    'WILDLIFE': 'wildlife',
    'FISHERIES': 'fisheries',
    'INST': 'institute',
    'CNTR': 'center',
    'LAB': 'laboratory',
    'DEANS': 'dean',
    'GENL': 'general',
    'INFECT': 'infectious',
    'DIS': 'disease',
    'ALLERGY': 'allergy',
    'MEMBRANE': 'membrane',
}

def expand_dept(dept):
    # Remove prefixes and deduplicate pipe-separated values
    parts = dept.split('|')
    dept = parts[0]  # take first value if duplicated
    dept = dept.replace('MED:', '').replace('VM:', '').strip().lower()
    # Remove parenthetical notes like (SAC)
    import re
    dept = re.sub(r'\(.*?\)', '', dept).strip()
    words = dept.split()
    expanded = [DEPT_EXPANSION.get(w.upper(), w) for w in words]
    return ' '.join(expanded)

def field_similarity_rapidfuzz(data):
    print("Running: 'field_similarity_rapidfuzz'...")
    scores = []

    for ucmps_field, scopus_field in data:
        scopus_clean = expand_dept(scopus_field).lower()
        
        ucmps_fields = [f.strip().lower() for f in ucmps_field.split(',')]
        ucmps_fields = [f.replace('(all)', '').replace('(miscellaneous)', '').strip() 
                       for f in ucmps_fields]
        ucmps_fields = [f for f in ucmps_fields if f]

        if not ucmps_fields or not scopus_clean:
            scores.append(0.0)
            continue

        best = process.extractOne(scopus_clean, ucmps_fields, scorer=fuzz.token_set_ratio)
        score = best[1] / 100.0 if best else 0.0
        scores.append(score)

    print("Completed running 'field_similarity_rapidfuzz'")
    return scores

#Research Area Similarity Combination
#@Brief: This section will combine the institution matching from the 
#research_area_similarity function and the fuzz ratio.
# #nltk_score is a 1xn list. So 

def field_similarity_feature(data):
    print("Running field_similarity_feature")
    field_similarity = []
    nltk_scores = research_area_similarity(data)

    for (ucpms_field, scopus_field), nltk_score in zip(data, nltk_scores):

        fuzzy_raw = fuzz.token_set_ratio(
            ucpms_field,
            scopus_field,
            processor=utils.default_process
        )

        if fuzzy_raw >= 80:
            fuzzy_score = 1
        elif fuzzy_raw >= 60:
            fuzzy_score = 0.5
        else:
            fuzzy_score = 0

        field_similarity.append(fuzzy_score + nltk_score)
    print("Completed running field_similarity_feature")
    return field_similarity




##################INSTITUTION FEATURE##########################
#@Brief: This function checks for similarity between the institutions listed in UCPMS and SCOPUS.
#        It will match ROR IDs when available. 
#        It should not be weighed as heavily as the name features, since institutions can change 
#        over time, but it is still a strong signal when it is present and matches.

def institution_similarity(data):
    print(f"Running: 'Institution similarity'...")
    davis_affiliations = [

    # Core institution variants
    "University of California Davis", "Univ California Davis", "UC-Davis", "UCD",
    "UC Davis Campus", "UC Davis Main Campus",

    # Location-based signals
    "Davis, CA", "Davis California", "Sacramento, CA", "Sacramento California",

    # Major departments (high-frequency in publications)
    "Department of Statistics", "Department of Computer Science",
    "Department of Mathematics", "Department of Physics",
    "Department of Chemistry", "Department of Biological Sciences",
    "Department of Molecular and Cellular Biology",
    "Department of Neurobiology, Physiology and Behavior",
    "Department of Biomedical Engineering",
    "Department of Mechanical and Aerospace Engineering",
    "Department of Civil and Environmental Engineering",
    "Department of Electrical and Computer Engineering",
    "Department of Chemical Engineering",
    "Department of Materials Science and Engineering",

    # Social sciences / humanities
    "Department of Economics", "Department of Sociology",
    "Department of Political Science", "Department of Psychology",
    "Department of Anthropology", "Department of History",
    "Department of English",

    # Health + clinical departments
    "Department of Internal Medicine", "Department of Surgery",
    "Department of Pediatrics", "Department of Pathology and Laboratory Medicine",
    "Department of Radiology", "Department of Neurology",
    "Department of Psychiatry and Behavioral Sciences",
    "Department of Public Health Sciences",

    # Additional institutes / centers (commonly cited)
    "Center for Mind and Brain",
    "Center for Neuroscience",
    "UC Davis Alzheimer’s Disease Center",
    "UC Davis Comprehensive Cancer Center",
    "MIND Institute",
    "Center for Data Science and Artificial Intelligence Research",
    "CeDAR",
    "Institute for Transportation Studies",
    "ITS Davis",
    "John Muir Institute of the Environment",
    "Tahoe Environmental Research Center",
    "Center for Watershed Sciences",
    "One Health Institute",
    "Global Disease Biology",
    "Foods for Health Institute",
    "Air Quality Research Center",
    "Energy and Efficiency Institute",
    "Western Cooling Efficiency Center",

    # Ag / environmental / extension
    "UC Cooperative Extension",
    "Cooperative Extension",
    "Agricultural Experiment Station",

    # Labs / facilities
    "Bodega Marine Laboratory",
    "Crocker Nuclear Laboratory",

    # Health system variants (expand)
    "UC Davis Health Sacramento",
    "UC Davis Medical Group",
    "UC Davis Hospital",
    "UCD Health System",

    # Common abbreviations authors use
    "UC Davis Dept",
    "UCD Dept",
    "UCD School of Medicine",
    "UCD School of Veterinary Medicine"
]
    
    affiliations = []  

    for scopus_institution, ucmps_department in data:
        affiliated = 0

        #Checking general location
        for text in [scopus_institution, ucmps_department]:
            if not text:
                continue
            if (("Davis" in text or "Sacramento" in text)
                and ("University" in text or "Department" in text or "School" in text)):
                affiliated += 0.5
        
            #Direct Matching
            if text in davis_affiliations:
                affiliated += 1
            
        #Fuzzy matching
        match, score, _ = process.extractOne(ucmps_department, davis_affiliations, 
                                             scorer=fuzz.token_set_ratio,processor=utils.default_process)
        if score > 85:
            affiliated += 0.5

        affiliations.append(affiliated)
    print(f"Completed running: 'Institution similarity'")
    return affiliations


# UCPMS-Linkage : Author Disambiguation Pipeline for Scholarly Impact Reporting

## Overview

This project automates author identity verification between the University of California Publication Management System (UCPMS) and SCOPUS. The system was developed during a UC Davis Library internship to address errors caused by ambiguous author names, duplicate SCOPUS profiles, name formatting inconsistencies, and institutional affiliation changes.

Accurate author identification is essential for institutional reporting, faculty publication tracking, bibliometric analyses, and university rankings. Manual verification of thousands of records is time-intensive, motivating the development of a machine learning–based author disambiguation workflow.

## Problem

The UCPMS retrieves publication records from SCOPUS using author profiles. However, author matching frequently fails because:

* Multiple researchers may share similar or identical names.
* Authors may publish under different name formats or initials.
* Researchers often accumulate multiple SCOPUS IDs over their careers.
* Institutional affiliations change over time.
* Publication metadata varies across systems.

These issues create incorrect publication assignments and reduce the reliability of institutional reporting.

## Solution

I developed a supervised machine learning pipeline that predicts whether a SCOPUS profile belongs to a given UC Davis researcher.

The pipeline performs:

1. API retrieval of author metadata
2. Data cleaning and normalization
3. Feature engineering
4. Model inference
5. Confidence-based review prioritization

The system was trained on approximately 2,600 manually verified author-profile decisions and evaluated on a held-out test set.

## Feature Engineering

The model combines multiple similarity signals:

### Name Features

* Name normalization and formatting correction
* Initial-versus-full-name matching
* Token-level name disambiguation
* Sequence similarity scoring

### Research Area Features

* Department and research field similarity
* RapidFuzz fuzzy matching
* WordNet semantic similarity measures
* Domain-specific abbreviation expansion

### Institutional Features

* UC Davis affiliation detection
* Department matching
* Fuzzy institutional similarity scoring

## Models Evaluated

### Logistic Regression

Used to generate interpretable probability estimates and confidence scores.

### Gradient Boosting Classifier

Used as the primary prediction model due to superior classification performance and its ability to capture nonlinear feature interactions.

## Results

### Best Gradient Boosting Performance

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 86%    |
| Weighted F1     | 85%    |
| Match Precision | 82–83% |
| Match Recall    | 84–85% |

Performance was achieved after iterative feature engineering, hyperparameter tuning, and validation across multiple experimental runs.

## Human-in-the-Loop Review System

Rather than fully automating decisions, the final workflow prioritizes records for manual review.

### High-Priority Review

* Logistic Regression and Gradient Boosting disagree.

### Medium-Priority Review

* Models agree, but Logistic Regression confidence is below threshold.

### Optional Review

* Lower-confidence predictions requiring additional verification.

This approach captures the majority of model errors while substantially reducing manual workload.

## Impact

The final system was designed to assist verification of more than 9,000 publication records.

Estimated impact:

* Reduced projected verification time from approximately 37 weeks of manual review to days of targeted verification.
* Created a reproducible and scalable author identity validation workflow.
* Improved reliability of publication attribution for UC Davis reporting and scholarly impact assessments.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* RapidFuzz
* SCOPUS APIs
* CSV/JSON data pipelines

## Future Improvements

Potential areas for future work include:

* ORCID integration
* Additional publication-title similarity features
* Jaro–Winkler and Jaccard similarity metrics
* Automated hyperparameter optimization
* Probability calibration
* Expanded training datasets
* Enhanced handling of uncertain ("inconclusive") cases

## Author

Chelsea Momoh

Statistics, University of California, Davis

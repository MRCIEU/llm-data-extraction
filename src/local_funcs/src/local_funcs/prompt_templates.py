metadataprompt = {
    "role": "user",
    "content": """What are the exposures, outcomes in this abstract? If there are multiple exposures or outcomes, provide them all. If there are no exposures or outcomes, provide an empty list. Also categorize the exposures and outcomes into the following groups using the exact category names provided: 
- molecular
- socioeconomic
- environmental
- behavioural
- anthropometric
- clinical measures
- infectious disease
- neoplasm
- disease of the blood and blood-forming organs
- metabolic disease
- mental disorder
- disease of the nervous system
- disease of the eye and adnexa
- disease of the ear and mastoid process
- disease of the circulatory system
- disease of the digestive system
- disease of the skin and subcutaneous tissue
- disease of the musculoskeletal system and connective tissue
- disease of the genitourinary system
If an exposure or outcome does not fit into any of these groups, specify "Other". 

List the analytical methods used in the abstract. Match the methods to the following list of exact method names. If a method is used that is not in the list, specify "Other" and also provide the name of the method. The list of methods is as follows:
- two-sample mendelian randomization
- multivariable mendelian randomization
- colocalization
- network mendelian randomization
- triangulation
- reverse mendelian randomization
- one-sample mendelian randomization
- negative controls
- sensitivity analysis
- non-linear mendelian randomization
- within-family mendelian randomization

Provide a description of the population(s) on which the study described in the abstract was based.

Provide your answer in strict pretty JSON format using exactly the format as the example output and without markdown code blocks. Any error messages and explanations must be included in the JSON output with the key "metainformation".
""",
}

resultsprompt = {
    "role": "user",
    "content": """
List all of the results in the abstract, with each entry comprising: exposure, outcome, beta, units, odds ratio, hazard ratio, 95% confidence interval, standard error, and P-value. If any of these fields is missing, substitute them with "null". Add a field called "direction" which describes whether the exposure "increases" or "decreases" the outcome. 
Provide your answer in strict pretty JSON format using exactly the format as the example output and without markdown code blocks. You must only include values explicitly written in the abstract. Any error messages and explanations must be included in the JSON output with the key "resultsinformation". 

""",
}


metadataexample = {
    "role": "assistant",
    "content": """This is an example output in JSON format: 
    { "metadata": {
    "exposures": [
    {
        "id": "1",
        "trait": "Particulate matter 2.5",
        "category": "Environmental"
    },
    {
        "id": "2",
        "trait": "Type 2 diabetes",
        "category": "metabolic disease"
    },
    {
        "id": "3",
        "trait": "Body mass index",
        "category": "Anthropometric"
    }
    ],
    "outcomes": [
    {
        "id": "1",
        "trait": "Forced expiratory volume in 1 s",
        "category": "Clinical measure"
    },
    {
        "id": "2",
        "trait": "Forced vital capacity",
        "category": "Clinical measure"
    },
    {
        "id": "3",
        "trait": "Gastroesophageal reflux disease",
        "category": "disease of the digestive system"
    },
    {
        "id": "4",
        "trait": "Non-alcoholic fatty liver disease (NAFLD)",
        "category": "disease of the digestive system"
    }
    ],
    "methods": ["two-sample mendelian randomization", "multivariable mendelian randomization", "colocalisation", "network mendelian randomization"],
    "population": ["European men", "Breast cancer patients", "African-Americans"],
    "metainformation": {
        "error": "No information on population is provided in abstract",
        "explanation": "Some methods do not match those listed in the prompt"
    }
    }
    }
    """,
}

resultsexample = {
    "role": "assistant",
    "content": """This is an example output in JSON format: 
    {
    "results": [
        {
            "exposure": "Particulate matter 2.5"},
            "outcome": "Forced expiratory volume in 1 s"
            "beta": 0.154,
            "units": "mmHg",
            "hazard ratio": null,
            "odds ratio": null,
            "95% CI": [0.101,0.215],
            "SE": 0.102,
            "P-value": 0.0015,
            "Direction": "increases"
        },
        {
            "exposure": "Body mass index"},
            "outcome": "Gastroesophageal reflux disease"
            "beta": null,
            "units": null,
            "hazard ratio": null,
            "odds ratio": 1.114,
            "95% CI": [1.021,1.314],
            "SE": null,
            "P-value": 0.0157,
            "Direction": "increases"
        },
        {
            "exposure": "Body mass index"},
            "outcome": "Non-alcoholic fatty liver disease (NAFLD)"
            "beta": null,
            "units": null,
            "hazard ratio": null,
            "odds ratio": null,
            "95% CI": [null,null],
            "SE": null,
            "P-value": null,
            "Direction": "increases"
        }
    ]
    "resultsinformation": {
        "error": "No results provided in abstract",
        "explanation": "P-values were string, not numeric values"
    }
    }
    """,
}

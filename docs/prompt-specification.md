## metadata message

```
{'content': 'You are a data scientist responsible for extracting accurate '
             'information from research papers. You answer each question with '
             'a single JSON string.',
'role': 'system'},
```

```
 {'content': '\n'
             '                This is an abstract from a Mendelian '
             'randomization study.\n'
             '                    "Alcohol consumption significantly impacts '
             'disease burden and has been linked to various diseases in '
             'observational studies. However, comprehensive meta-analyses '
             'using Mendelian randomization (MR) to examine drinking patterns '
             'are limited. We aimed to evaluate the health risks of alcohol '
             'use by integrating findings from MR studies. A thorough search '
             'was conducted for MR studies focused on alcohol exposure. We '
             'utilized two sets of instrumental variables-alcohol consumption '
             'and problematic alcohol use-and summary statistics from the '
             'FinnGen consortium R9 release to perform de novo MR analyses. '
             'Our meta-analysis encompassed 64 published and 151 de novo MR '
             'analyses across 76 distinct primary outcomes. Results show that '
             'a genetic predisposition to alcohol consumption, independent of '
             'smoking, significantly correlates with a decreased risk of '
             "Parkinson's disease, prostate hyperplasia, and rheumatoid "
             'arthritis. It was also associated with an increased risk of '
             'chronic pancreatitis, colorectal cancer, and head and neck '
             'cancers. Additionally, a genetic predisposition to problematic '
             'alcohol use is strongly associated with increased risks of '
             'alcoholic liver disease, cirrhosis, both acute and chronic '
             'pancreatitis, and pneumonia. Evidence from our MR study supports '
             'the notion that alcohol consumption and problematic alcohol use '
             'are causally associated with a range of diseases, predominantly '
             'by increasing the risk."   \n'
             '                    ',
  'role': 'user'},
```

```
 {'content': 'This is an example output in JSON format: \n'
             '    { "metadata": {\n'
             '    "exposures": [\n'
             '    {\n'
             '        "id": "1",\n'
             '        "trait": "Particulate matter 2.5",\n'
             '        "category": "Environmental"\n'
             '    },\n'
             '    {\n'
             '        "id": "2",\n'
             '        "trait": "Type 2 diabetes",\n'
             '        "category": "metabolic disease"\n'
             '    },\n'
             '    {\n'
             '        "id": "3",\n'
             '        "trait": "Body mass index",\n'
             '        "category": "Anthropometric"\n'
             '    }\n'
             '    ],\n'
             '    "outcomes": [\n'
             '    {\n'
             '        "id": "1",\n'
             '        "trait": "Forced expiratory volume in 1 s",\n'
             '        "category": "Clinical measure"\n'
             '    },\n'
             '    {\n'
             '        "id": "2",\n'
             '        "trait": "Forced vital capacity",\n'
             '        "category": "Clinical measure"\n'
             '    },\n'
             '    {\n'
             '        "id": "3",\n'
             '        "trait": "Gastroesophageal reflux disease",\n'
             '        "category": "disease of the digestive system"\n'
             '    },\n'
             '    {\n'
             '        "id": "4",\n'
             '        "trait": "Non-alcoholic fatty liver disease (NAFLD)",\n'
             '        "category": "disease of the digestive system"\n'
             '    }\n'
             '    ],\n'
             '    "methods": ["two-sample mendelian randomization", '
             '"multivariable mendelian randomization", "colocalisation", '
             '"network mendelian randomization"],\n'
             '    "population": ["European men", "Breast cancer patients", '
             '"African-Americans"],\n'
             '    "metainformation": {\n'
             '        "error": "No information on population is provided in '
             'abstract",\n'
             '        "explanation": "Some methods do not match those listed '
             'in the prompt"\n'
             '    }\n'
             '    }\n'
             '    }\n'
             '    ',
  'role': 'assistant'},
```

```
 {'content': 'What are the exposures, outcomes in this abstract? If there are '
             'multiple exposures or outcomes, provide them all. If there are '
             'no exposures or outcomes, provide an empty list. Also categorize '
             'the exposures and outcomes into the following groups using the '
             'exact category names provided: \n'
             '- molecular\n'
             '- socioeconomic\n'
             '- environmental\n'
             '- behavioural\n'
             '- anthropometric\n'
             '- clinical measures\n'
             '- infectious disease\n'
             '- neoplasm\n'
             '- disease of the blood and blood-forming organs\n'
             '- metabolic disease\n'
             '- mental disorder\n'
             '- disease of the nervous system\n'
             '- disease of the eye and adnexa\n'
             '- disease of the ear and mastoid process\n'
             '- disease of the circulatory system\n'
             '- disease of the digestive system\n'
             '- disease of the skin and subcutaneous tissue\n'
             '- disease of the musculoskeletal system and connective tissue\n'
             '- disease of the genitourinary system\n'
             'If an exposure or outcome does not fit into any of these groups, '
             'specify "Other". \n'
             '\n'
             'List the analytical methods used in the abstract. Match the '
             'methods to the following list of exact method names. If a method '
             'is used that is not in the list, specify "Other" and also '
             'provide the name of the method. The list of methods is as '
             'follows:\n'
             '- two-sample mendelian randomization\n'
             '- multivariable mendelian randomization\n'
             '- colocalization\n'
             '- network mendelian randomization\n'
             '- triangulation\n'
             '- reverse mendelian randomization\n'
             '- one-sample mendelian randomization\n'
             '- negative controls\n'
             '- sensitivity analysis\n'
             '- non-linear mendelian randomization\n'
             '- within-family mendelian randomization\n'
             '\n'
             'Provide a description of the population(s) on which the study '
             'described in the abstract was based.\n'
             '\n'
             'Provide your answer in strict pretty JSON format using exactly '
             'the format as the example output and without markdown code '
             'blocks. Any error messages and explanations must be included in '
             'the JSON output with the key "metainformation".\n',
  'role': 'user'}]
```

## results message

```
{'content': 'You are a data scientist responsible for extracting accurate '
             'information from research papers. You answer each question with '
             'a single JSON string.',
  'role': 'system'}
```

```
{'content': '\n'
             '                This is an abstract from a Mendelian '
             'randomization study.\n'
             '                    "Alcohol consumption significantly impacts '
             'disease burden and has been linked to various diseases in '
             'observational studies. However, comprehensive meta-analyses '
             'using Mendelian randomization (MR) to examine drinking patterns '
             'are limited. We aimed to evaluate the health risks of alcohol '
             'use by integrating findings from MR studies. A thorough search '
             'was conducted for MR studies focused on alcohol exposure. We '
             'utilized two sets of instrumental variables-alcohol consumption '
             'and problematic alcohol use-and summary statistics from the '
             'FinnGen consortium R9 release to perform de novo MR analyses. '
             'Our meta-analysis encompassed 64 published and 151 de novo MR '
             'analyses across 76 distinct primary outcomes. Results show that '
             'a genetic predisposition to alcohol consumption, independent of '
             'smoking, significantly correlates with a decreased risk of '
             "Parkinson's disease, prostate hyperplasia, and rheumatoid "
             'arthritis. It was also associated with an increased risk of '
             'chronic pancreatitis, colorectal cancer, and head and neck '
             'cancers. Additionally, a genetic predisposition to problematic '
             'alcohol use is strongly associated with increased risks of '
             'alcoholic liver disease, cirrhosis, both acute and chronic '
             'pancreatitis, and pneumonia. Evidence from our MR study supports '
             'the notion that alcohol consumption and problematic alcohol use '
             'are causally associated with a range of diseases, predominantly '
             'by increasing the risk."   \n'
             '                    ',
  'role': 'user'}
```

```
{'content': 'This is an example output in JSON format: \n'
             '    {\n'
             '    "results": [\n'
             '        {\n'
             '            "exposure": "Particulate matter 2.5"},\n'
             '            "outcome": "Forced expiratory volume in 1 s"\n'
             '            "beta": 0.154,\n'
             '            "units": "mmHg",\n'
             '            "hazard ratio": null,\n'
             '            "odds ratio": null,\n'
             '            "95% CI": [0.101,0.215],\n'
             '            "SE": 0.102,\n'
             '            "P-value": 0.0015,\n'
             '            "Direction": "increases"\n'
             '        },\n'
             '        {\n'
             '            "exposure": "Body mass index"},\n'
             '            "outcome": "Gastroesophageal reflux disease"\n'
             '            "beta": null,\n'
             '            "units": null,\n'
             '            "hazard ratio": null,\n'
             '            "odds ratio": 1.114,\n'
             '            "95% CI": [1.021,1.314],\n'
             '            "SE": null,\n'
             '            "P-value": 0.0157,\n'
             '            "Direction": "increases"\n'
             '        },\n'
             '        {\n'
             '            "exposure": "Body mass index"},\n'
             '            "outcome": "Non-alcoholic fatty liver disease '
             '(NAFLD)"\n'
             '            "beta": null,\n'
             '            "units": null,\n'
             '            "hazard ratio": null,\n'
             '            "odds ratio": null,\n'
             '            "95% CI": [null,null],\n'
             '            "SE": null,\n'
             '            "P-value": null,\n'
             '            "Direction": "increases"\n'
             '        }\n'
             '    ]\n'
             '    "resultsinformation": {\n'
             '        "error": "No results provided in abstract",\n'
             '        "explanation": "P-values were string, not numeric '
             'values"\n'
             '    }\n'
             '    }\n'
             '    ',
  'role': 'assistant'}
```

```
{'content': '\n'
             'List all of the results in the abstract, with each entry '
             'comprising: exposure, outcome, beta, units, odds ratio, hazard '
             'ratio, 95% confidence interval, standard error, and P-value. If '
             'any of these fields is missing, substitute them with "null". Add '
             'a field called "direction" which describes whether the exposure '
             '"increases" or "decreases" the outcome. \n'
             'Provide your answer in strict pretty JSON format using exactly '
             'the format as the example output and without markdown code '
             'blocks. You must only include values explicitly written in the '
             'abstract. Any error messages and explanations must be included '
             'in the JSON output with the key "resultsinformation". \n'
             '\n',
  'role': 'user'}
```

# categories

## trait (exposure / outcome) categories

or "Other"

```
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
```

## method categories

or "Other"

```
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
```

## results schema

```
- exposure
- outcome
- beta
- units
- odds ratio
- hazard ratio
- 95% confidence interval
- standard error
- and P-value
```

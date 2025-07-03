#!/usr/bin/env python3

import re

def parse_azphewas_phenotype(phenotype_text: str) -> str:
    """
    Parse AzPheWAS phenotype text to extract meaningful clinical terms.
    """
    parts = phenotype_text.split('#')
    
    if len(parts) == 1:
        # No # separator, return as is (NMR entries)
        return phenotype_text.strip()
    
    elif len(parts) == 2:
        # Format: CODE#DESCRIPTION
        code, description = parts
        
        # Check if it's a "Source of report" entry
        if description.startswith("Source of report of"):
            # Extract condition name after ICD code
            # Pattern: "Source of report of [ICD_CODE] [CONDITION_NAME]"
            match = re.match(r'Source of report of ([A-Z]\d+(?:\.\d+)?)\s+(.+)', description)
            if match:
                return match.group(2).strip()
            else:
                # Fallback: remove "Source of report of" and any leading ICD-like codes
                clean_desc = description.replace("Source of report of", "").strip()
                # Remove leading ICD codes (like A01, B15, etc.)
                clean_desc = re.sub(r'^[A-Z]\d+(?:\.\d+)?\s+', '', clean_desc)
                return clean_desc.strip()
        else:
            # Regular CODE#DESCRIPTION format
            return description.strip()
    
    elif len(parts) == 3:
        # Format: CODE#ICD_CODE#DESCRIPTION (41202 series)
        return parts[2].strip()
    
    else:
        # More than 3 parts, take the last part
        return parts[-1].strip()

# Test cases
test_cases = [
    "120000#Ever had osteoarthritis affecting one or more joints e g hip knee shoulder",
    "130003#Source of report of A01 typhoid and paratyphoid fevers", 
    "41202#E871#Hypo-osmolality and hyponatraemia",
    "41202#C10#Malignant neoplasm of oropharynx",
    "41202#Block Z40-Z54#Z40-Z54 Persons encountering health services for specific procedures and health care",
    "30600#Albumin",
    "NMR Cholesterol to Total Lipids in HDL percentage"
]

print("Testing AzPheWAS phenotype parsing:")
print("="*60)

for test_case in test_cases:
    parsed = parse_azphewas_phenotype(test_case)
    print(f"Original: {test_case}")
    print(f"Parsed:   {parsed}")
    print() 
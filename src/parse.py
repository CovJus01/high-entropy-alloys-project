import pandas as pd
import re
from collections import defaultdict


# Load the CSV file
input_file = "../data/High Entropy Alloy Properties.csv"
output_file = "../data/High_Entropy_Alloy_Parsed.csv"

df = pd.read_csv(input_file)

# Extract all unique elements from the FORMULA column
element_counts = defaultdict(int)
struct_counts = defaultdict(int)
method_counts = defaultdict(int)
formula_regex = re.compile(r"([A-Z][a-z]*)([\d\.]*)")

for formula in df["FORMULA"].dropna():
    matches = formula_regex.findall(formula)
    for element, _ in matches:
        element_counts[element] += 1

for struct in df["PROPERTY: Microstructure"].dropna():
    struct_counts[struct] += 1


for method in df["PROPERTY: Processing method"].dropna():
    method_counts[method] += 1

# List of unique elements
unique_elements = sorted(element_counts.keys())
unique_structs = sorted(struct_counts.keys())
unique_methods = sorted(method_counts.keys())

# Create new columns for each element
df = df.assign(**{element: 0.0 for element in unique_elements})
df = df.assign(**{struct: 0.0 for struct in unique_structs})
df = df.assign(**{method: 0.0 for method in unique_methods})
df = df.assign(**{struct: 0.0 for struct in ["BCC_col", "FCC_col", "other"]})
df = df.assign(**{test: 0.0 for test in ["C_test", "T_test"]})

# Populate the element columns
for idx, formula in df["FORMULA"].dropna().items():
    element_values = {el: 0.0 for el in unique_elements}
    matches = formula_regex.findall(formula)
    for element, value in matches:
        element_values[element] = float(value) if value else 1.0  # Default to 1 if no number is given
    for element, value in element_values.items():
        df.at[idx, element] = value

for idx, structure in df["PROPERTY: Microstructure"].dropna().items():
        df.at[idx, structure] = 1.0


for idx, method in df["PROPERTY: Processing method"].dropna().items():
        df.at[idx, method] = 1.0

# Encode Structure
for idx, struct in df["PROPERTY: BCC/FCC/other"].dropna().items():
    if(struct == "BCC"):
        df.at[idx, "BCC_col"] = 1.0
    elif(struct == "FCC"):
        df.at[idx, "FCC_col"] = 1.0
    elif(struct == "other"):
        df.at[idx, "other"] = 1.0

# Encode Type of test
for idx, test in df["PROPERTY: Type of test"].dropna().items():
    if(test == "C"):
        df.at[idx, "C_test"] = 1.0
    elif(test == "T"):
        df.at[idx, "T_test"] = 1.0

# Save the modified DataFrame
df.to_csv(output_file, index=False)

print(f"Processed file saved as: {output_file}")

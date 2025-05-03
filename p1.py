from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import uuid
import numpy as np

# ----------------------------- User Inputs --------------------------------
stream = input("Enter your stream (Arts, Commerce, Engineering, Medical): ").strip().lower()
branches = input("Enter 5 preferred branches (comma-separated): ").split(",")
branches = [b.strip() for b in branches[:5]]  # Limit to 5
your_location = input("Enter your location: ")
family_income = int(input("Enter your family income: "))
inc_Scale = float(input("Enter your flexibility scale (1-5) for fees: "))
pref_Scale = float(input("Enter your flexibility scale (1-5) for location: "))
rank_scale = float(input("Enter your flexibility scale (1-5) for rank: "))
rating_scale = float(input("Enter your flexibility scale (1-5) for rating: "))
preferred_locations = input("Enter your preferred locations (comma-separated): ").split(",")
preferred_locations = [loc.strip() for loc in preferred_locations]

# ----------------------------- Dataset Mapping --------------------------------
stream_datasets = {
    'arts': {'college': 'careerpeek.ai/Arts.csv', 'skills': 'careerpeek.ai/skillsarts.csv'},
    'commerce': {'college': 'careerpeek.ai/commerce.csv', 'skills': 'careerpeek.ai/skillscommerce.csv'},
    'engineering': {'college': 'careerpeek.ai/engineering.csv', 'skills': 'careerpeek.ai/skillseng.csv'},
    'medical': {'college': 'careerpeek.ai/Medical.csv', 'skills': 'careerpeek.ai/skillsmedi.csv'}
}

if stream not in stream_datasets:
    print(f"Invalid stream. Please choose from: {', '.join(stream_datasets.keys())}")
    exit()

college_file = stream_datasets[stream]['college']
skills_file = stream_datasets[stream]['skills']
branch_file = 'careerpeek.ai/branch.csv'  # New branch.csv file

# ----------------------------- Load College Data ----------------------------------
try:
    df_colleges = pd.read_csv(college_file)
    df_colleges['First_Year_Fees'] = df_colleges['First_Year_Fees'].astype(str).str.replace(',', '').astype(float)
    df_colleges['Cutoff'] = pd.to_numeric(df_colleges['Cutoff'], errors='coerce')
    df_colleges['Income_Diff'] = abs(df_colleges['First_Year_Fees'] - family_income)

    # Normalization
    scaler = MinMaxScaler()
    columns_to_normalize = ['First_Year_Fees', 'Average Package', 'Highest Package', 'National Rank']
    df_colleges[columns_to_normalize] = scaler.fit_transform(df_colleges[columns_to_normalize])
except FileNotFoundError:
    print(f"College dataset {college_file} not found.")
    exit()

# ----------------------------- Load Skills Data ----------------------------------
try:
    df_skills = pd.read_csv(skills_file)
except FileNotFoundError:
    print(f"Skills dataset {skills_file} not found.")
    exit()

# ----------------------------- Load Branch Data ----------------------------------
try:
    df_branch = pd.read_csv(branch_file)
except FileNotFoundError:
    print(f"Branch dataset {branch_file} not found.")
    exit()

# Update df_skills with user-entered branches from branch.csv
degree_col = 'Degree' if stream == 'arts' else 'Name' if stream == 'commerce' else 'Branch' if stream == 'engineering' else 'Degree/Doctor'
for branch in branches:
    branch_match = df_branch[df_branch['Branch'].str.lower() == branch.lower()]
    if not branch_match.empty:
        cost = branch_match['Skill_Acquisition_Cost_INR'].iloc[0]
        trend = branch_match['Future_Trend_1_10'].iloc[0]
        
        # Check if branch exists in df_skills
        if branch.lower() in df_skills[degree_col].str.lower().values:
            # Update existing branch
            df_skills.loc[df_skills[degree_col].str.lower() == branch.lower(), 'Skill_Acquisition_Cost_INR'] = cost
            df_skills.loc[df_skills[degree_col].str.lower() == branch.lower(), 'Future_Trend_1_10'] = trend
        else:
            # Append new branch
            new_row = {
                degree_col: branch,
                'Skill_Acquisition_Cost_INR': cost,
                'Future_Trend_1_10': trend
            }
            df_skills = pd.concat([df_skills, pd.DataFrame([new_row])], ignore_index=True)

# Process skill acquisition cost
if stream in ['arts', 'engineering']:
    df_skills['Skill_Acquisition_Cost_INR'] = df_skills['Skill_Acquisition_Cost_INR'].apply(
        lambda x: np.mean([float(i) for i in x.split('-')]) if isinstance(x, str) and '-' in x else float(x)
    )
elif stream == 'commerce':
    df_skills['Skill_Acquisition_Cost_INR'] = df_skills['Skill_Acquisition_Cost_INR']  # Already in correct format
elif stream == 'medical':
    df_skills['Skill_Acquisition_Cost_INR'] = df_skills['Skill_Acquisition_Cost_INR'].apply(
        lambda x: 2000000 if x == 20 else float(x)
    )

# Normalize skill costs and trends
df_skills['Normalized_Cost'] = scaler.fit_transform(df_skills[['Skill_Acquisition_Cost_INR']])
df_skills['Future_Trend_1_10'] = df_skills.get('Future_Trend_1_10', df_skills.get('Future Trend (1-10)', 1))
df_skills['Heuristic'] = df_skills['Future_Trend_1_10'] / 10  # Scale to 0-1

# ----------------------------- A* for Colleges ---------------------------
def calculate_cost_matrix_colleges(df):
    df_filtered = df  # No branch filtering

    geolocator = Nominatim(user_agent=f"distance_calculator_{uuid.uuid4()}", timeout=5)
    location_coords = {}
    for loc in preferred_locations:
        try:
            location = geolocator.geocode(loc)
            if location:
                location_coords[loc] = (location.latitude, location.longitude)
        except:
            pass
        time.sleep(1)

    distance_matrix = []
    for index, row in df_filtered.iterrows():
        row_distances = []
        try:
            college_loc = geolocator.geocode(row['City'])
            if college_loc is None:
                row_distances = [None] * len(preferred_locations)
            else:
                college_coords = (college_loc.latitude, college_loc.longitude)
                for loc in preferred_locations:
                    if loc in location_coords:
                        dist = geodesic(location_coords[loc], college_coords).kilometers
                        row_distances.append(dist)
                    else:
                        row_distances.append(None)
        except:
            row_distances = [None] * len(preferred_locations)
        distance_matrix.append(row_distances)
        time.sleep(1)

    ds = pd.DataFrame(distance_matrix, columns=preferred_locations, index=df_filtered.index)
    ds.fillna(ds.max().max(), inplace=True)
    ds_normalized = pd.DataFrame(scaler.fit_transform(ds), columns=preferred_locations, index=df_filtered.index)
    df_filtered['Avg_Location_Distance'] = ds_normalized.mean(axis=1)
    df_filtered['Normalized_Fees'] = scaler.fit_transform(df_filtered[['Income_Diff']])

    # Normalize cutoff inversely
    df_filtered['Cutoff'] = df_filtered['Cutoff'].fillna(df_filtered['Cutoff'].max())
    df_filtered['Cutoff_Score'] = (df_filtered['Cutoff'].max() - df_filtered['Cutoff']) / (df_filtered['Cutoff'].max() - df_filtered['Cutoff'].min())

    # Calculate heuristic
    df_filtered['Heuristic'] = (rating_scale * df_filtered['Rating'] + rank_scale * df_filtered['National Rank'] + df_filtered['Average Package']) * 0.04

    cost_matrix = []
    for idx, row in df_filtered.iterrows():
        costy = (
            pref_Scale * row['Avg_Location_Distance'] +
            inc_Scale * row['Normalized_Fees'] +
            0.2 * (1 - row['Cutoff_Score'])
        )
        if costy <= row['Heuristic']:
            costy = row['Heuristic'] + 0.1

        cost_matrix.append([row['College'], round(costy, 4), round(row['Heuristic'], 4)])

    cost_matrix = pd.DataFrame(cost_matrix, columns=['College', 'Cost', 'Heuristic'])
    cost_matrix['Total'] = cost_matrix['Cost'] + cost_matrix['Heuristic']
    sorted_colleges = cost_matrix.sort_values(by='Total').head(10)

    if not sorted_colleges.empty:
        print(f"\n🔍 Top 10 Colleges for {stream.capitalize()} Based on Cost and Heuristic:\n")
        print(sorted_colleges)

    return sorted_colleges

# ----------------------------- A* for Skills ---------------------------
def calculate_cost_matrix_skills(df):
    df_filtered = df  # No branch filtering

    cost_matrix = []
    for idx, row in df_filtered.iterrows():
        costy = row['Normalized_Cost']
        heuristic = row['Heuristic']
        if costy <= heuristic:
            costy = heuristic + 0.1
        degree_col = 'Degree' if stream == 'arts' else 'Name' if stream == 'commerce' else 'Branch' if stream == 'engineering' else 'Degree/Doctor'
        cost_matrix.append([
            row[degree_col],
            round(costy, 4),
            round(heuristic, 4),
            row['Skill_Acquisition_Cost_INR']
        ])

    cost_matrix = pd.DataFrame(cost_matrix, columns=['Degree/Skill', 'Cost', 'Heuristic', 'Acquisition_Cost_INR'])
    cost_matrix['Total'] = cost_matrix['Cost'] + cost_matrix['Heuristic']
    sorted_skills = cost_matrix.sort_values(by='Total').head(5)

    total_cost = sorted_skills['Acquisition_Cost_INR'].sum()

    if not sorted_skills.empty:
        print(f"\n🔍 Top 5 Skills for {stream.capitalize()} Based on Cost and Heuristic (Total Cost: {total_cost:.2f} INR):\n")
        print(sorted_skills[['Degree/Skill', 'Cost', 'Heuristic', 'Total', 'Acquisition_Cost_INR']])

    return sorted_skills, total_cost

# ----------------------------- Main Logic ---------------------------------
print(f"\n=== Recommendations for Stream: {stream.capitalize()} ===\n")

# College recommendations
top_colleges = calculate_cost_matrix_colleges(df_colleges)

# Skill recommendations
top_skills, total_cost = calculate_cost_matrix_skills(df_skills)

print(f"\nPreferred Branches (for reference): {', '.join(branches)}")


from flask import Flask, render_template, request, jsonify
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import uuid
import numpy as np
import logging
from heapq import heappush, heappop
import json

app = Flask(__name__)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "careerpeek.ai/data/"

stream_datasets = {
    'arts': {'college': f'{DATA_PATH}Arts.csv', 'skills': f'{DATA_PATH}skillsarts.csv'},
    'commerce': {'college': f'{DATA_PATH}commerce.csv', 'skills': f'{DATA_PATH}skillscommm.csv'},
    'engineering': {'college': f'{DATA_PATH}engineering.csv', 'skills': f'{DATA_PATH}skillseng.csv'},
    'medical': {'college': f'{DATA_PATH}Medical.csv', 'skills': f'{DATA_PATH}skillsmedi.csv'}
}
branch_file = f'{DATA_PATH}branch.csv'

skill_branch_mapping = {
    'AI/ML': ['Machine Learning', 'Deep Learning', 'Data Analysis', 'Python', 'TensorFlow'],
    'Computer Science': ['Software Development', 'Cloud Computing', 'Networking', 'Java', 'JavaScript'],
    'Accounting': ['Financial Reporting', 'Taxation', 'Auditing', 'Excel', 'QuickBooks'],
    'Finance': ['Investment Analysis', 'Risk Management', 'Financial Modeling', 'Excel', 'Bloomberg Terminal'],
    'Medicine': ['Clinical Practice', 'Surgical Skills', 'Diagnostics', 'Medical Research', 'EMR Systems'],
}

def generate_career_path(streams, branches, colleges, skills, num_paths=3):
    logger.debug(f"Generating {num_paths} diverse career paths with A*")
    
    open_set = []
    visited = set()
    valid_paths = []
    used_combinations = set()
    
    for stream in streams:
        start_node = (stream, 0, [stream])
        heappush(open_set, (0, start_node))
    
    while open_set and len(valid_paths) < num_paths:
        _, (current_node, total_cost, path) = heappop(open_set)
        
        if len(path) == 4:
            combination = (path[1], path[2])  
            if combination not in used_combinations:
                logger.debug(f"Found diverse path: {path}, total_cost={total_cost}")
                valid_paths.append({
                    'stream': path[0],
                    'branch': path[1],
                    'college': path[2],
                    'skill': path[3],
                    'total_cost': total_cost
                })
                used_combinations.add(combination)
            continue
        
        if len(path) == 1:  
            next_nodes = [(b, 0, 0) for b in branches]
        elif len(path) == 2:  
            next_nodes = [(c['College'], c['Cost'], c['Heuristic']) for c in colleges[:3]]
        elif len(path) == 3:
            branch = path[1]
            relevant_skills = [s for s in skills if s['Degree/Skill'] in skill_branch_mapping.get(branch, [])]
            if not relevant_skills:
                relevant_skills = skills
            next_nodes = [(s['Skills'] or s['Degree/Skill'], s['Cost'], s['Heuristic']) for s in relevant_skills]
        
        for next_node, cost, heuristic in next_nodes:
            node_key = (len(path), next_node)
            if node_key in visited:
                continue
            visited.add(node_key)
            
            edge_cost = cost + heuristic
            new_total_cost = total_cost + edge_cost
            
            new_path = path + [next_node]
            
            remaining_layers = 3 - len(path)
            heuristic_estimate = remaining_layers * 0.1
            priority = new_total_cost + heuristic_estimate
            
            heappush(open_set, (priority, (next_node, new_total_cost, new_path)))
    
    if not valid_paths:
        logger.warning("No valid career paths found")
        return [{
            'stream': streams[0] if streams else 'N/A',
            'branch': branches[0] if branches else 'N/A',
            'college': colleges[0]['College'] if colleges else 'N/A',
            'skill': skills[0]['Skills'] or skills[0]['Degree/Skill'] if skills else 'N/A',
            'total_cost': 0
        }]
    
    logger.debug(f"Returning {len(valid_paths)} diverse career paths")
    return valid_paths

@app.route('/')
def index():
    logger.debug("Rendering p2.html")
    return render_template('p2.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        logger.debug("Received POST request to /recommend")
        data = request.form
        streams = [s.strip().lower() for s in data['streams'].split(',')]
        branches = [b.strip() for b in data['branches'].split(',')][:5]
        your_location = data['location']
        family_income = int(data['family_income'])
        inc_scale = float(data['inc_scale'])
        pref_scale = float(data['pref_scale'])
        rank_scale = float(data['rank_scale'])
        rating_scale = float(data['rating_scale'])
        preferred_locations = [loc.strip() for loc in data['preferred_locations'].split(',')]
        logger.debug(f"Form data: streams={streams}, branches={branches}, location={your_location}, income={family_income}")

        valid_streams = [s for s in streams if s in stream_datasets]
        if not valid_streams:
            logger.error(f"Invalid streams: {streams}")
            return jsonify({'error': f"Invalid streams. Choose from: {', '.join(stream_datasets.keys())}"}), 400

        all_colleges = []
        all_skills = []
        for stream in valid_streams:
            college_file = stream_datasets[stream]['college']
            skills_file = stream_datasets[stream]['skills']
            logger.debug(f"Loading datasets for {stream}: college={college_file}, skills={skills_file}")

            try:
                logger.debug(f"Loading college data for {stream}")
                df_colleges = pd.read_csv(college_file)
                df_colleges['First_Year_Fees'] = pd.to_numeric(df_colleges['First_Year_Fees'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                df_colleges['Cutoff'] = pd.to_numeric(df_colleges['Cutoff'], errors='coerce').fillna(df_colleges['Cutoff'].max() if df_colleges['Cutoff'].notna().any() else 0)
                df_colleges['Average_Package'] = pd.to_numeric(df_colleges['Average_Package'], errors='coerce').fillna(0)
                df_colleges['Highest_Package'] = pd.to_numeric(df_colleges['Highest_Package'], errors='coerce').fillna(0)
                df_colleges['National Rank'] = pd.to_numeric(df_colleges['National Rank'], errors='coerce').fillna(0)
                df_colleges['Rating'] = pd.to_numeric(df_colleges['Rating'], errors='coerce').fillna(0)
                df_colleges['Income_Diff'] = abs(df_colleges['First_Year_Fees'] - family_income)
                scaler = MinMaxScaler()
                columns_to_normalize = ['First_Year_Fees', 'Average_Package', 'Highest_Package', 'National Rank']
                df_colleges[columns_to_normalize] = scaler.fit_transform(df_colleges[columns_to_normalize])
                all_colleges.append(df_colleges)
                logger.debug(f"College data loaded for {stream}")
            except FileNotFoundError:
                logger.error(f"College dataset not found: {college_file}")
                return jsonify({'error': f"College dataset {college_file} not found."}), 500
            except KeyError as e:
                logger.error(f"Missing column in college dataset: {e}")
                return jsonify({'error': f"Missing column in college dataset: {e}"}), 500

            try:
                logger.debug(f"Loading skills data for {stream}")
                df_skills = pd.read_csv(skills_file, dtype={'Cost': str, 'Branch': str}, skiprows=0)
                df_skills['Branch'] = df_skills['Branch'].str.strip().fillna('N/A')
                df_skills['Skills'] = df_skills['Skills'].str.strip().fillna('N/A')
                df_skills['Heuristics'] = pd.to_numeric(df_skills['Heuristics'], errors='coerce').fillna(1)
                logger.debug(f"Raw Cost values for {stream}:\n{df_skills[['Branch', 'Cost', 'Skills']].head().to_string()}")
                all_skills.append(df_skills)
                logger.debug(f"Skills data loaded for {stream}")
            except FileNotFoundError:
                logger.error(f"Skills dataset not found: {skills_file}")
                return jsonify({'error': f"Skills dataset {skills_file} not found."}), 500
            except KeyError as e:
                logger.error(f"Missing column in skills dataset: {e}")
                return jsonify({'error': f"Missing column in skills dataset: {e}"}), 500

        try:
            logger.debug("Loading branch data")
            df_branch = pd.read_csv(branch_file, dtype={'Cost': str, 'Branch': str})
            df_branch['Branch'] = df_branch['Branch'].str.strip().fillna('N/A')
            df_branch['Heuristics'] = pd.to_numeric(df_branch['Heuristics'], errors='coerce').fillna(1)
            logger.debug(f"Raw Cost values for {stream}:\n{df_skills[['Branch', 'Cost', 'Skills']].head().to_string()}")
            all_skills.append(df_skills)
            logger.debug(f"Skills data loaded for {stream}")
        except FileNotFoundError:
                logger.error(f"Skills dataset not found: {skills_file}")
                return jsonify({'error': f"Skills dataset {skills_file} not found."}), 500
        except KeyError as e:
                logger.error(f"Missing column in skills dataset: {e}")
                return jsonify({'error': f"Missing column in skills dataset: {e}"}), 500

        try:
            logger.debug("Loading branch data")
            df_branch = pd.read_csv(branch_file, dtype={'Cost': str, 'Branch': str})
            df_branch['Branch'] = df_branch['Branch'].str.strip().fillna('N/A')
            df_branch['Heuristics'] = pd.to_numeric(df_branch['Heuristics'], errors='coerce').fillna(1)
            logger.debug(f"Branch data:\n{df_branch[['Branch', 'Cost']].head().to_string()}")
            logger.debug("Branch data loaded successfully")
        except FileNotFoundError:
            logger.error(f"Branch dataset not found: {branch_file}")
            return jsonify({'error': f"Branch dataset {branch_file} not found."}), 500
        except KeyError as e:
            logger.error(f"Missing column in branch dataset: {e}")
            return jsonify({'error': f"Missing column in branch dataset: {e}"}), 500

        logger.debug("Updating skills with branch data")
        degree_col = 'Branch'
        for stream, df_skills in zip(valid_streams, all_skills):
            for branch in branches:
                branch_match = df_branch[df_branch['Branch'].str.lower() == branch.lower()]
                if not branch_match.empty:
                    cost = branch_match['Cost'].iloc[0]
                    heuristic = branch_match['Heuristics'].iloc[0]
                    logger.debug(f"Matched branch {branch}: Cost={cost}, Heuristic={heuristic}")
                    if branch.lower() in df_skills[degree_col].str.lower().values:
                        df_skills.loc[df_skills[degree_col].str.lower() == branch.lower(), 'Cost'] = cost
                        df_skills.loc[df_skills[degree_col].str.lower() == branch.lower(), 'Heuristics'] = heuristic
                        logger.debug(f"Updated cost for {branch} in df_skills")
                    else:
                        new_row = {
                            degree_col: branch,
                            'Cost': cost,
                            'Heuristics': heuristic,
                            'Skills': ';'.join(skill_branch_mapping.get(branch, ['N/A']))
                        }
                        df_skills = pd.concat([df_skills, pd.DataFrame([new_row])], ignore_index=True)
                        logger.debug(f"Added new row for {branch}: {new_row}")
                else:
                    logger.warning(f"No branch match found for: {branch}")
            all_skills[valid_streams.index(stream)] = df_skills
        logger.debug("Skills updated with branch data")

        logger.debug("Processing skill acquisition costs")
        for stream, df_skills in zip(valid_streams, all_skills):
            if stream in ['arts', 'engineering']:
                df_skills['Cost'] = df_skills['Cost'].apply(
                    lambda x: np.mean([float(i) for i in x.split('-')]) if isinstance(x, str) and '-' in x and x not in ['N/A', 'Skill Acquisition Cost (INR)'] else float(x) if pd.notnull(x) and x not in ['N/A', 'Skill Acquisition Cost (INR)'] and x.strip() else 100000
                )
            elif stream == 'commerce':
                df_skills['Cost'] = df_skills['Cost'].apply(
                    lambda x: np.mean([float(i) for i in x.split('-')]) if isinstance(x, str) and '-' in x and x not in ['N/A', 'Skill Acquisition Cost (INR)'] else float(x) if pd.notnull(x) and x not in ['N/A', 'Skill Acquisition Cost (INR)'] and x.strip() else 100000
                )
            elif stream == 'medical':
                df_skills['Cost'] = df_skills['Cost'].apply(
                    lambda x: 2000000 if x == '20' or (pd.notnull(x) and str(x).strip() == '20') else float(x) * 100000 if pd.notnull(x) and x not in ['N/A', 'Skill Acquisition Cost (INR)'] and x.strip() else 100000
                )
            logger.debug(f"Processed costs for {stream}:\n{df_skills[['Branch', 'Cost', 'Skills']].head().to_string()}")
        logger.debug("Skill costs processed")

        logger.debug("Normalizing skill costs and trends")
        scaler = MinMaxScaler()
        for df_skills in all_skills:
            df_skills['Normalized_Cost'] = scaler.fit_transform(df_skills[['Cost']])
            df_skills['Normalized_Cost'] = df_skills['Normalized_Cost'].fillna(0)
            df_skills['Heuristic'] = df_skills['Heuristics'] / 10
            df_skills['Heuristic'] = df_skills['Heuristic'].fillna(0.1)
            logger.debug(f"Normalized skills for {stream}:\n{df_skills[['Branch', 'Cost', 'Normalized_Cost', 'Heuristic', 'Skills']].head().to_string()}")
            if df_skills['Cost'].isna().any() or df_skills['Normalized_Cost'].isna().any():
                logger.warning(f"NaN values detected in skills: {df_skills[df_skills['Cost'].isna() | df_skills['Normalized_Cost'].isna()]}")
        print("----------------------------------SKILLS----------------------------------------")
        print(df_skills.head())
        logger.debug("Normalization complete")

        def calculate_cost_matrix_colleges(df):
            logger.debug("Starting college A* calculation")
            df_filtered = df
            geolocator = Nominatim(user_agent=f"distance_calculator_{uuid.uuid4()}", timeout=5)
            location_coords = {}
            for loc in preferred_locations:
                logger.debug(f"Geocoding location: {loc}")
                try:
                    location = geolocator.geocode(loc)
                    if location:
                        location_coords[loc] = (location.latitude, location.longitude)
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Geocoding failed for {loc}: {e}")
            logger.debug(f"Geocoded locations: {location_coords}")

            distance_matrix = []
            for index, row in df_filtered.iterrows():
                row_distances = []
                logger.debug(f"Geocoding college city: {row['City']}")
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
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Geocoding failed for {row['City']}: {e}")
                    row_distances = [None] * len(preferred_locations)
                distance_matrix.append(row_distances)
            logger.debug("Distance matrix calculated")

            ds = pd.DataFrame(distance_matrix, columns=preferred_locations, index=df_filtered.index)
            ds.fillna(ds.max().max() if ds.max().max() > 0 else 1000, inplace=True)
            ds_normalized = pd.DataFrame(scaler.fit_transform(ds), columns=preferred_locations, index=df_filtered.index)
            df_filtered['Avg_Location_Distance'] = ds_normalized.mean(axis=1)
            
            df_filtered['Normalized_Fees'] = scaler.fit_transform(df_filtered[['Income_Diff']])
            
            df_filtered['Cutoff'] = df_filtered['Cutoff'].fillna(df_filtered['Cutoff'].max())
            if df_filtered['Cutoff'].max() == df_filtered['Cutoff'].min():
                df_filtered['Cutoff_Score'] = 0
            else:
                df_filtered['Cutoff_Score'] = (df_filtered['Cutoff'].max() - df_filtered['Cutoff']) / (df_filtered['Cutoff'].max() - df_filtered['Cutoff'].min())
            # explain2
            df_filtered['Heuristic'] = (rating_scale * df_filtered['Rating'] + rank_scale * df_filtered['National Rank'] + df_filtered['Average_Package']) * 0.04
            # Explain1
            cost_matrix = []
            for idx, row in df_filtered.iterrows():
                costy = (
                    pref_scale * row['Avg_Location_Distance'] +
                    inc_scale * row['Normalized_Fees'] +
                    0.2 * (1 - row['Cutoff_Score'])
                )
                if costy <= row['Heuristic']:
                    costy = row['Heuristic'] + 0.1
                cost_matrix.append([row['College'], round(costy, 4), round(row['Heuristic'], 4)])
            
            cost_matrix = pd.DataFrame(cost_matrix, columns=['College', 'Cost', 'Heuristic'])
            cost_matrix['Total'] = cost_matrix['Cost'] + cost_matrix['Heuristic']
            sorted_colleges = cost_matrix.sort_values(by='Total').head(10)
            logger.debug("College A* calculation complete")
            return sorted_colleges.to_dict('records')

        def calculate_cost_matrix_skills(df):
            logger.debug("Starting skills A* calculation")
            df_filtered = df.copy()
            cost_matrix = []
            for idx, row in df_filtered.iterrows():
                costy = row['Normalized_Cost']
                heuristic = row['Heuristic']
                acquisition_cost = float(row['Cost']) if pd.notnull(row['Cost']) and row['Cost'] != 0 else 100000
                if costy <= heuristic:
                    costy = heuristic + 0.1
                cost_matrix.append([
                    row['Branch'],
                    row.get('Skills', ''),
                    round(costy, 4),
                    round(heuristic, 4),
                    acquisition_cost
                ])
            
            cost_matrix = pd.DataFrame(cost_matrix, columns=['Degree/Skill', 'Skills', 'Cost', 'Heuristic', 'Acquisition_Cost_INR'])

            logger.debug(f"Skills cost matrix:\n{cost_matrix[['Degree/Skill', 'Acquisition_Cost_INR', 'Skills']].head().to_string()}")
            cost_matrix['Total'] = cost_matrix['Cost'] + cost_matrix['Heuristic']
            print("-----------------------------------------------------CostMatrix-----------------------------------------------------------------------------")
            print(cost_matrix.head())
            sorted_skills = cost_matrix.sort_values(by='Total', ascending=False).head(5)

            total_cost = (sorted_skills['Cost'] + sorted_skills['Heuristic']).sum()

            logger.debug(f"Skills A* calculation complete, total_cost={total_cost}")

            return sorted_skills.to_dict('records'), total_cost

        logger.debug("Running A* algorithms")
        top_colleges = []
        top_skills = []
        total_cost = 0
        for stream, df_colleges, df_skills in zip(valid_streams, all_colleges, all_skills):
            colleges = calculate_cost_matrix_colleges(df_colleges)
            skills, cost = calculate_cost_matrix_skills(df_skills)
            top_colleges.extend(colleges)
            top_skills.extend(skills)
            total_cost += cost
        logger.debug("A* algorithms complete")

        career_paths = generate_career_path(valid_streams, branches, top_colleges, top_skills, num_paths=3)

        response_data = {
            'colleges': top_colleges,
            'skills': top_skills,
            'total_cost': float(total_cost),
            'preferred_branches': branches,
            'career_paths': career_paths
        }
        logger.debug(f"JSON response prepared: {json.dumps(response_data, indent=2)}")

        logger.debug("Returning JSON response")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in /recommend: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
import json
import re
from fuzzywuzzy import fuzz, process
import pandas as pd
import os

class LocationMatcher:
    def __init__(self, knowledge_base_path=None):
        """Initialize the LocationMatcher with an optional knowledge base file path.
        
        Args:
            knowledge_base_path (str): Path to the knowledge base JSON or TXT file
        """
        self.locations = {}
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
    
    def load_knowledge_base(self, file_path):
        """Load location data from a knowledge base file.
        
        The file can be either:
        - JSON: [{"name": "Tai Po Metro Station", "lat": 22.444499, "lon": 114.170409}, ...]
        - TXT: "Tai Po Metro Station - 22.444499, 114.170409" (one entry per line)
        - CSV: The first column should be the name, second column latitude, third column longitude
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Knowledge base file not found: {file_path}")
            
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for location in data:
                    self.locations[location['name']] = (location['lat'], location['lon'])
        
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ' - ' in line and ',' in line:
                        name, coords = line.split(' - ', 1)
                        lat, lon = coords.split(',', 1)
                        try:
                            self.locations[name.strip()] = (float(lat.strip()), float(lon.strip()))
                        except ValueError:
                            print(f"Warning: Couldn't parse coordinates in line: {line}")
        
        elif file_path.endswith('.csv'):
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                # Assuming first column is name, second is lat, third is lon
                for _, row in df.iterrows():
                    columns = row.values
                    if len(columns) >= 3:
                        name = str(columns[0]).strip()
                        try:
                            lat = float(columns[1])
                            lon = float(columns[2])
                            self.locations[name] = (lat, lon)
                        except (ValueError, TypeError):
                            print(f"Warning: Could not parse coordinates for {name}")
            except Exception as e:
                print(f"Error processing CSV file: {e}")
                
        else:
            raise ValueError("Unsupported file format. Use .json, .txt, or .csv files")
            
        print(f"Loaded {len(self.locations)} locations from knowledge base")
    
    def add_locations_from_dataframe(self, df, name_col, lat_col, lon_col):
        """Add locations from a pandas DataFrame.
        
        Args:
            df (DataFrame): Pandas DataFrame containing location data
            name_col (str): Column name for location names
            lat_col (str): Column name for latitude
            lon_col (str): Column name for longitude
        """
        for _, row in df.iterrows():
            name = str(row[name_col])
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            self.locations[name] = (lat, lon)
        
        print(f"Added {len(df)} locations from DataFrame")
    
    def extract_location_from_text(self, text):
        """Extract location information from user input text.
        
        Args:
            text (str): User input text
            
        Returns:
            tuple: (lat, lon) if direct coordinates are found,
                  location_name if a location might be mentioned,
                  None if no location information is found
        """
        # Check for direct coordinate pattern: "latitude, longitude" or similar
        coord_pattern = r'(\d+\.\d+)\s*,\s*(\d+\.\d+)'
        coord_matches = re.findall(coord_pattern, text)
        
        if coord_matches:
            try:
                lat, lon = float(coord_matches[0][0]), float(coord_matches[0][1])
                return (lat, lon)
            except ValueError:
                pass
        
        # Look for location names in our knowledge base with fuzzy matching
        potential_locations = []
        
        # Extract all possible location-like phrases
        # This is a simplistic approach - you might need more sophisticated NLP
        # Common patterns: "in {location}", "near {location}", "at {location}"
        location_patterns = [
            r'(?:in|at|near|around|by)\s+([A-Za-z0-9\s]+(?:\s+[Ss]tation)?)',
            r'([A-Za-z0-9\s]+(?:\s+[Ss]tation))',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            potential_locations.extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        return potential_locations if potential_locations else None
    
    def find_best_location_match(self, location_text, threshold=70):
        """Find the best match for a location name using fuzzy matching.
        
        Args:
            location_text (str or list): Location name or list of potential location names
            threshold (int): Minimum score threshold for fuzzy matching (0-100)
            
        Returns:
            tuple: (matched_name, (lat, lon), match_score) or None if no match above threshold
        """
        if not self.locations:
            print("Warning: No locations loaded in knowledge base")
            return None
            
        if not location_text:
            return None
            
        if isinstance(location_text, list):
            # Try each potential location
            best_match = None
            best_score = 0
            best_name = None
            
            for loc in location_text:
                match, score = process.extractOne(loc, list(self.locations.keys()))
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = match
                    best_name = loc
            
            if best_match:
                print(f"Best match for '{best_name}' is '{best_match}' with score {best_score}")
                return (best_match, self.locations[best_match], best_score)
            return None
        else:
            # Direct match for a single location name
            match, score = process.extractOne(location_text, list(self.locations.keys()))
            if score >= threshold:
                print(f"Matched '{location_text}' to '{match}' with score {score}")
                return (match, self.locations[match], score)
            return None
    
    def process_user_query(self, query_text):
        """Process a user query to extract and match location information.
        
        Args:
            query_text (str): The user's query text
            
        Returns:
            dict: {
                'input_type': 'coordinates'|'location_name'|'unknown',
                'matched_location': name of matched location (if found),
                'coordinates': (lat, lon) tuple,
                'original_input': original location text or coordinates,
                'query': original query with coordinates inserted (if match found),
                'match_score': the matching score (0-100) if a location name was matched
            }
        """
        result = {
            'input_type': 'unknown',
            'matched_location': None,
            'coordinates': None,
            'original_input': None,
            'query': query_text,
            'match_score': None
        }
        
        location_info = self.extract_location_from_text(query_text)
        
        if location_info is None:
            return result
            
        # Case 1: Direct coordinates in the query
        if isinstance(location_info, tuple) and len(location_info) == 2:
            result['input_type'] = 'coordinates'
            result['coordinates'] = location_info
            result['original_input'] = f"{location_info[0]}, {location_info[1]}"
            return result
            
        # Case 2: Location name(s) extracted
        best_match = self.find_best_location_match(location_info)
        if best_match:
            matched_name, coords, match_score = best_match
            result['input_type'] = 'location_name'
            result['matched_location'] = matched_name
            result['coordinates'] = coords
            result['match_score'] = match_score
            
            # For a list of potential locations, find which one matched
            if isinstance(location_info, list):
                original_input = None
                matching_threshold = 70
                # Find the original input that best matches the matched name
                original_input = None
                for loc in location_info:
                    if fuzz.token_sort_ratio(loc, matched_name) >= matching_threshold:
                        original_input = loc
                        break
                
                # If we can't find a good match, use the first one
                if not original_input and location_info:
                    original_input = location_info[0]
                
                result['original_input'] = original_input
            else:
                result['original_input'] = location_info
            
            # Replace the location name with coordinates in the original query
            if result['original_input']:
                modified_query = query_text.replace(
                    result['original_input'], 
                    f"{coords[0]}, {coords[1]}"
                )
                result['query'] = modified_query
            
        return result

# Direct API usage
if __name__ == "__main__":
    # Load from a knowledge base file
    matcher = LocationMatcher("locations.json")  # Can also use .txt or .csv files
    
    # You can also add locations programmatically
    matcher.locations["Custom Location"] = (22.123456, 114.123456)
    
    # Process a single query
    query = "What is the traffic like near Tai Po?"
    result = matcher.process_user_query(query)
    
    if result['coordinates']:
        print(f"Matched '{result['matched_location']}' -> {result['coordinates']}")
        print(f"Modified query: {result['query']}")
        print(f"Match score: {result['match_score']}%")
    else:
        print("No location match found in query")
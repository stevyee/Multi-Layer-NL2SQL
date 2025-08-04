from langchain_community.chat_models import ChatOpenAI

def extract_location_keyword(user_input):
    """
    Uses an AI agent to extract the location keyword from the user input.
    Returns the extracted location name(s).
    """
    # Initialize the AI model
    llm = ChatOpenAI(model="o1-mini", temperature=1)
    
    # Create a prompt that extracts location names
    prompt = f"""
Your task is to identify and extract any location names from the following text.
Focus only on geographic locations like cities, towns, neighborhoods, stations, buildings, etc.
Return ONLY the extracted location name(s) without any additional text or explanations.
If there are multiple locations, separate them with a semicolon.
If no location is found, respond with "NO_LOCATION_FOUND".

Text: "{user_input}"

Location(s):
"""
    
    # Call the AI agent
    response = llm.invoke(prompt).content.strip()
    
    # Process the response
    if response == "NO_LOCATION_FOUND":
        return None
    
    # Split by semicolon if multiple locations were returned
    locations = [loc.strip() for loc in response.split(';') if loc.strip()]
    
    print(f"Extracted location(s): {locations}")
    return locations
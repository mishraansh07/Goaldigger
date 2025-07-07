# team_aliases.py

team_name_aliases = {
    # Premier League (England)
    "manchester united": "Manchester United",
    "man utd": "Manchester United",
    "liverpool": "Liverpool",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "tottenham hotspur": "Tottenham Hotspur",
    "spurs": "Tottenham Hotspur",
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "aston villa": "Aston Villa",
    "newcastle united": "Newcastle United",
    "west ham united": "West Ham United",
    "west ham": "West Ham United",
    "everton": "Everton",
    "leicester city": "Leicester City",
    "brighton & hove albion": "Brighton & Hove Albion",
    "brighton": "Brighton & Hove Albion",
    "fulham": "Fulham",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "wolves": "Wolverhampton Wanderers",
    "crystal palace": "Crystal Palace",
    "brentford": "Brentford",
    "burnley": "Burnley",
    "sheffield united": "Sheffield United",
    "sheff utd": "Sheffield United",
    "luton town": "Luton Town",
    "nottingham forest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    "bournemouth": "AFC Bournemouth", # Often just "Bournemouth"
    "afc bournemouth": "AFC Bournemouth",
    "leeds united": "Leeds United", # If in the Premier League for a given season
    "sunderland": "Sunderland", # If in the Premier League for a given season
    "southampton": "Southampton", # If in the Premier League for a given season

    # La Liga (Spain)
    "real madrid": "Real Madrid",
    "barcelona": "Barcelona",
    "atletico madrid": "Atlético Madrid",
    "athletic bilbao": "Athletic Club de Bilbao", # Often just "Athletic" or "Athletic Bilbao"
    "athletic club": "Athletic Club de Bilbao",
    "sevilla": "Sevilla FC", # Often just "Sevilla"
    "real betis": "Real Betis",
    "valencia": "Valencia CF", # Often just "Valencia"
    "villarreal": "Villarreal CF", # Often just "Villarreal"
    "real sociedad": "Real Sociedad",
    "celta vigo": "RC Celta de Vigo", # Often just "Celta Vigo"
    "getafe": "Getafe CF",
    "rayo vallecano": "Rayo Vallecano",
    "osasuna": "CA Osasuna",
    "alaves": "Deportivo Alavés", # Often just "Alavés"
    "deportivo alaves": "Deportivo Alavés",
    "cadiz": "Cádiz CF",
    "mallorca": "RCD Mallorca",
    "granada": "Granada CF",
    "las palmas": "UD Las Palmas",
    "girona": "Girona FC", # Often just "Girona"
    "elche": "Elche CF", # Often just "Elche"
    "espanyol": "RCD Espanyol", # Often just "Espanyol"
    "almeria": "UD Almería", # Often just "Almería"
    "ud almeria": "UD Almería",
    "real valladolid": "Real Valladolid", # If in La Liga for a given season
    "levante": "Levante UD", # If in La Liga for a given season
    "oviedo": "Real Oviedo", # If in La Liga for a given season
    "leganes": "CD Leganés", # If in La Liga for a given season

    # Serie A (Italy)
    "juventus": "Juventus", # Verify in your DB (might be 'Juventus FC' etc.)
    "ac milan": "Milan", # Verify in your DB (might be 'A.C. Milan', 'AC Milan', 'Milan')
    "milan": "Milan", # Alias for AC Milan
    "inter milan": "Internazionale", # Verify in your DB (might be 'FC Internazionale', 'Inter')
    "inter": "Internazionale",
    "napoli": "Napoli",
    "roma": "AS Roma", # Often just "Roma"
    "as roma": "AS Roma",
    "lazio": "Lazio",
    "atalanta": "Atalanta",
    "fiorentina": "Fiorentina",
    "torino": "Torino",
    "udinese": "Udinese",
    "bologna": "Bologna",
    "sassuolo": "US Sassuolo Calcio", # Often just "Sassuolo"
    "us sassuolo": "US Sassuolo Calcio",
    "sampdoria": "Sampdoria", # If in Serie A for a given season
    "genoa": "Genoa",
    "verona": "Hellas Verona", # Often just "Verona"
    "hellas verona": "Hellas Verona",
    "lecce": "Lecce",
    "monza": "Monza",
    "cremonese": "Cremonese",
    "salernitana": "Salernitana", # If in Serie A for a given season
    "empoli": "Empoli",
    "cagliari": "Cagliari",
    "como": "Como 1907", # Often just "Como"
    "parma": "Parma Calcio 1913", # Often just "Parma"
    "venezia": "Venezia FC", # Often just "Venezia"
    "pisa": "Pisa SC", # If in Serie A for a given season

    # Bundesliga (Germany)
    "bayern munich": "FC Bayern München", # Example: Adjust if your DB uses different name
    "bayern munchen": "FC Bayern München",
    "borussia dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "rb leipzig": "RB Leipzig",
    "bayer leverkusen": "Bayer 04 Leverkusen",
    "leverkusen": "Bayer 04 Leverkusen",
    "eintracht frankfurt": "Eintracht Frankfurt",
    "freiburg": "SC Freiburg",
    "hoffenheim": "1899 Hoffenheim",
    "wolfsburg": "VfL Wolfsburg",
    "werder bremen": "SV Werder Bremen",
    "borussia monchengladbach": "Borussia Mönchengladbach",
    "monchengladbach": "Borussia Mönchengladbach",
    "koln": "1. FC Köln", # Often just "Köln"
    "mainz": "1. FSV Mainz 05",
    "augsburg": "FC Augsburg",
    "stuttgart": "VfB Stuttgart",
    "bochum": "VfL Bochum 1848",
    "darmstadt": "SV Darmstadt 98",
    "union berlin": "1. FC Union Berlin",
    "heidenheim": "1. FC Heidenheim 1846", # Often just "Heidenheim"
    "fc heidenheim": "1. FC Heidenheim 1846",
    "hamburg": "Hamburger SV", # If in Bundesliga for a given season
    "hamburger sv": "Hamburger SV",
    "schalke": "FC Schalke 04", # If in Bundesliga for a given season
    "fc schalke 04": "FC Schalke 04",
    "hertha bsc": "Hertha BSC", # If in Bundesliga for a given season
    "st pauli": "FC St. Pauli", # If in Bundesliga for a given season
    "fortuna dusseldorf": "Fortuna Düsseldorf", # If in Bundesliga for a given season
    "karlsruher sc": "Karlsruher SC", # If in Bundesliga for a given season
    "nurnberg": "1. FC Nürnberg", # If in Bundesliga for a given season
    "hannover 96": "Hannover 96", # If in Bundesliga for a given season
    "holstein kiel": "Holstein Kiel", # If in Bundesliga for a given season

    # Ligue 1 (France)
    "paris saint-germain": "Paris Saint-Germain",
    "psg": "Paris Saint-Germain",
    "reims": "Stade de Reims",
    "nice": "OGC Nice",
    "lens": "RC Lens",
    "lille": "LOSC Lille",
    "marseille": "Olympique de Marseille",
    "lyon": "Olympique Lyonnais",
    "rennes": "Stade Rennais FC",
    "monaco": "AS Monaco",
    "strasbourg": "RC Strasbourg Alsace",
    "montpellier": "Montpellier Hérault SC",
    "brest": "Stade Brestois 29",
    "le havre": "Le Havre AC",
    "metz": "FC Metz",
    "clermont": "Clermont Foot 63",
    "nantes": "FC Nantes", # Often just "Nantes"
    "toulouse": "Toulouse FC", # Often just "Toulouse"
    "auxerre": "AJ Auxerre", # Often just "Auxerre"
    "aj auxerre": "AJ Auxerre",
    "angers": "Angers SCO", # Often just "Angers"
    "angers sco": "Angers SCO",
    "lorient": "FC Lorient", # Often just "Lorient"
    "saint-etienne": "AS Saint-Étienne", # If in Ligue 1 for a given season
    "as saint-etienne": "AS Saint-Étienne",
    "bordeaux": "FC Girondins de Bordeaux", # If in Ligue 1 for a given season
}

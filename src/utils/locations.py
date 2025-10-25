"""
Philippine Regions and Provinces
Data source: Philippine Statistics Authority (PSA)
"""

# Philippine regions with their provinces
PH_REGIONS = {
    "NCR - National Capital Region": [
        "Metro Manila"
    ],
    "CAR - Cordillera Administrative Region": [
        "Abra",
        "Apayao",
        "Benguet",
        "Ifugao",
        "Kalinga",
        "Mountain Province"
    ],
    "Region I - Ilocos Region": [
        "Ilocos Norte",
        "Ilocos Sur",
        "La Union",
        "Pangasinan"
    ],
    "Region II - Cagayan Valley": [
        "Batanes",
        "Cagayan",
        "Isabela",
        "Nueva Vizcaya",
        "Quirino"
    ],
    "Region III - Central Luzon": [
        "Aurora",
        "Bataan",
        "Bulacan",
        "Nueva Ecija",
        "Pampanga",
        "Tarlac",
        "Zambales"
    ],
    "Region IV-A - CALABARZON": [
        "Batangas",
        "Cavite",
        "Laguna",
        "Quezon",
        "Rizal"
    ],
    "Region IV-B - MIMAROPA": [
        "Marinduque",
        "Occidental Mindoro",
        "Oriental Mindoro",
        "Palawan",
        "Romblon"
    ],
    "Region V - Bicol Region": [
        "Albay",
        "Camarines Norte",
        "Camarines Sur",
        "Catanduanes",
        "Masbate",
        "Sorsogon"
    ],
    "Region VI - Western Visayas": [
        "Aklan",
        "Antique",
        "Capiz",
        "Guimaras",
        "Iloilo",
        "Negros Occidental"
    ],
    "Region VII - Central Visayas": [
        "Bohol",
        "Cebu",
        "Negros Oriental",
        "Siquijor"
    ],
    "Region VIII - Eastern Visayas": [
        "Biliran",
        "Eastern Samar",
        "Leyte",
        "Northern Samar",
        "Samar",
        "Southern Leyte"
    ],
    "Region IX - Zamboanga Peninsula": [
        "Zamboanga del Norte",
        "Zamboanga del Sur",
        "Zamboanga Sibugay"
    ],
    "Region X - Northern Mindanao": [
        "Bukidnon",
        "Camiguin",
        "Lanao del Norte",
        "Misamis Occidental",
        "Misamis Oriental"
    ],
    "Region XI - Davao Region": [
        "Davao de Oro",
        "Davao del Norte",
        "Davao del Sur",
        "Davao Occidental",
        "Davao Oriental"
    ],
    "Region XII - SOCCSKSARGEN": [
        "Cotabato",
        "Sarangani",
        "South Cotabato",
        "Sultan Kudarat"
    ],
    "Region XIII - Caraga": [
        "Agusan del Norte",
        "Agusan del Sur",
        "Dinagat Islands",
        "Surigao del Norte",
        "Surigao del Sur"
    ],
    "BARMM - Bangsamoro Autonomous Region in Muslim Mindanao": [
        "Basilan",
        "Lanao del Sur",
        "Maguindanao",
        "Sulu",
        "Tawi-Tawi"
    ]
}

# Flatten list of all provinces
ALL_PROVINCES = []
for provinces in PH_REGIONS.values():
    ALL_PROVINCES.extend(provinces)

# Sort alphabetically
ALL_PROVINCES.sort()

# Get list of region names
REGION_NAMES = list(PH_REGIONS.keys())

# Province coordinates (latitude, longitude)
# Data source: Approximate provincial capitals/centers
PROVINCE_COORDINATES = {
    # NCR
    "Metro Manila": [14.5995, 120.9842],
    
    # CAR
    "Abra": [17.5969, 120.7708],
    "Apayao": [18.0119, 121.1710],
    "Benguet": [16.4023, 120.5960],
    "Ifugao": [16.8333, 121.1833],
    "Kalinga": [17.4000, 121.4667],
    "Mountain Province": [17.0833, 121.0167],
    
    # Region I
    "Ilocos Norte": [18.1667, 120.7167],
    "Ilocos Sur": [17.5797, 120.3850],
    "La Union": [16.6159, 120.3209],
    "Pangasinan": [15.8950, 120.2863],
    
    # Region II
    "Batanes": [20.4500, 121.9700],
    "Cagayan": [18.2500, 121.7167],
    "Isabela": [16.9667, 121.8167],
    "Nueva Vizcaya": [16.3300, 121.1400],
    "Quirino": [16.2667, 121.5333],
    
    # Region III
    "Aurora": [15.7535, 121.6259],
    "Bataan": [14.6417, 120.4818],
    "Bulacan": [14.7942, 120.8794],
    "Nueva Ecija": [15.5784, 120.9842],
    "Pampanga": [15.0794, 120.6200],
    "Tarlac": [15.4754, 120.5964],
    "Zambales": [15.5085, 119.9730],
    
    # Region IV-A
    "Batangas": [13.7565, 121.0583],
    "Cavite": [14.2456, 120.8782],
    "Laguna": [14.2691, 121.4113],
    "Quezon": [14.0160, 122.1390],
    "Rizal": [14.6037, 121.3084],
    
    # Region IV-B
    "Marinduque": [13.4767, 121.9033],
    "Occidental Mindoro": [13.1000, 120.7667],
    "Oriental Mindoro": [13.0000, 121.4500],
    "Palawan": [9.8349, 118.7384],
    "Romblon": [12.5778, 122.2690],
    
    # Region V
    "Albay": [13.1391, 123.7311],
    "Camarines Norte": [14.1386, 122.7614],
    "Camarines Sur": [13.5291, 123.3483],
    "Catanduanes": [13.7046, 124.2461],
    "Masbate": [12.3700, 123.6300],
    "Sorsogon": [12.9742, 124.0067],
    
    # Region VI
    "Aklan": [11.8984, 122.0790],
    "Antique": [11.5667, 121.9500],
    "Capiz": [11.5492, 122.7500],
    "Guimaras": [10.5928, 122.6314],
    "Iloilo": [10.7202, 122.5621],
    "Negros Occidental": [10.6593, 122.9794],
    
    # Region VII
    "Bohol": [9.8500, 124.1435],
    "Cebu": [10.3157, 123.8854],
    "Negros Oriental": [9.3167, 123.3000],
    "Siquijor": [9.2000, 123.5833],
    
    # Region VIII
    "Biliran": [11.5833, 124.4667],
    "Eastern Samar": [11.5000, 125.5167],
    "Leyte": [11.0000, 124.8333],
    "Northern Samar": [12.4167, 124.7500],
    "Samar": [11.7500, 125.0000],
    "Southern Leyte": [10.3333, 125.1667],
    
    # Region IX
    "Zamboanga del Norte": [8.5500, 123.3167],
    "Zamboanga del Sur": [7.8381, 123.2966],
    "Zamboanga Sibugay": [7.7667, 122.4667],
    
    # Region X
    "Bukidnon": [8.0542, 124.9292],
    "Camiguin": [9.1733, 124.7300],
    "Lanao del Norte": [8.0000, 123.8333],
    "Misamis Occidental": [8.5000, 123.8333],
    "Misamis Oriental": [8.5000, 124.6167],
    
    # Region XI
    "Davao de Oro": [7.6667, 125.9167],
    "Davao del Norte": [7.5614, 125.6531],
    "Davao del Sur": [6.7731, 125.3300],
    "Davao Occidental": [6.0900, 125.6100],
    "Davao Oriental": [7.3167, 126.5500],
    
    # Region XII
    "Cotabato": [7.2167, 124.2833],
    "Sarangani": [5.9267, 125.0967],
    "South Cotabato": [6.3372, 124.8453],
    "Sultan Kudarat": [6.5000, 124.4167],
    
    # Region XIII
    "Agusan del Norte": [8.9458, 125.5319],
    "Agusan del Sur": [8.5561, 125.9736],
    "Dinagat Islands": [10.1286, 125.6050],
    "Surigao del Norte": [9.7833, 125.4833],
    "Surigao del Sur": [8.6167, 126.0500],
    
    # BARMM
    "Basilan": [6.4333, 121.9833],
    "Lanao del Sur": [7.8231, 124.4164],
    "Maguindanao": [6.9417, 124.4094],
    "Sulu": [6.0500, 121.0000],
    "Tawi-Tawi": [5.1333, 119.9500],
}

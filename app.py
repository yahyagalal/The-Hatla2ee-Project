import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import os

# URL of the ZIP file in your GitHub repository
url = "https://github.com/yahyagalal/The-Hatla2ee-Project/raw/main/hatla2ee_model.zip"

# Path where the ZIP file will be downloaded
zip_path = "./hatla2ee_model.zip"

# Path where the model file will be extracted
model_path = "./hatla2ee_model.joblib"


# Path where the cleaned dataset CSV will be extracted
cleaned_data_path = "./hatla2ee_cleaned_make_model_only.csv"

# Download the ZIP file
if not os.path.exists(zip_path):
#    st.write("Downloading model...")
    response = requests.get(url)
    with open(zip_path, "wb") as file:
        file.write(response.content)

# Unzip the file
if not os.path.exists(model_path):
#    st.write("Unzipping model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

# Load the model
#st.write("Loading model...")
model = joblib.load(model_path)
if hasattr(model, '__version__'):
    st.write("Scikit-learn version used to save the model:", model.__version__)
else:
    st.write("Scikit-learn version attribute not found in the model.")
try:
    model = joblib.load(model_path)
    st.write("Model loaded successfully!")
    st.write(f"Model type: {type(model)}")
    st.write(f"Model details: {model}")
except Exception as e:
    st.write(f"Error loading model: {e}")
#st.write("Model loaded successfully!")
st.write(model)

scaler=joblib.load('./scaler.joblib')

# Function to make predictions


# Create the Streamlit app
st.title("Used Car Price Prediction by Yahia Galal")



df_cleaned=pd.read_csv(cleaned_data_path)


#st.write(df_cleaned.head())






possible_car_models=['1 Series', '116', '118', '125', '14', '156', '180', '180B', '190', '2',
       '2 Series', '200', '2008', '206', '207', '207 SW', '208', '2105', '218',
       '218 i', '230', '240', '244', '250', '280', '3', '300', '3008', '301', '306',
       '307', '308', '308 sw', '316', '318', '320', '323', '325', '328', '330',
       '330S', '335', '340', '350', '360', '405', '406', '407', '408', '418',
       '420', '5', '5 Series', '500', '500 X', '5008', '500C', '504', '505',
       '508', '520', '523', '525', '528', '530', '535', '6', '605', '607',
       '620', '7', '740', '750', 'A 10', 'A 150', 'A 180', 'A 200', 'A 210',
       'A 35 AMG', 'A1', 'A11', 'A113', 'A15', 'A25', 'A3', 'A30 Shine', 'A4',
       'A5', 'A516', 'A6', 'A620', 'APV', 'Acadia', 'Accent', 'Accent HCI',
       'Accent RB', 'Accord', 'Aeolus A30', 'Alsvin', 'Altea', 'Altea XL',
       'Alto', 'Arona', 'Arrizo 5', 'Arrizo 5 pro', 'Arteon', 'Astra', 'Astro',
       'Ateca', 'Atos', 'Attrage', 'Auris', 'Avalanche', 'Avante', 'Avanza',
       'Avensis', 'Aveo', 'Azera', 'B 150', 'B 160', 'B 180', 'B 200', 'B70', 'Baleno', 'Bayon', 'Beetle', 'Belta', 'Benni',
       'Berlingo', 'Bluebird', 'Bora', 'Bravo', 'C 180', 'C 200', 'C 200 AMG',
       'C 230', 'C 240', 'C 250', 'C 280', 'C 300', 'C Class', 'C Elysee',
       'C-HR', 'C2', 'C3', 'C3 Aircross', 'C31', 'C4', 'C4 Grand Picasso',
       'C4 Picasso', 'C4X', 'C5', 'C5 Aircross', 'CC', 'CLA 180', 'CLA 200',
       'CLA 250', 'CLC Class', 'CLK', 'CRV', 'CS 15', 'CS 35', 'CS 35 Plus',
       'CS 55', 'CS55 Plus', 'Caddy', 'Camaro', 'Camry', 'Caprice', 'Captiva', 'Captur', 'Carens', 'Carnival', 'Cascada', 'Cayenne S', 'Ceed',
       'Celerio', 'Cerato', 'Cerato Koup', 'Charade', 'Charger', 'Cherokee',
       'Ciaz', 'Civic', 'Ck', 'Ck2', 'Clio', 'Clubman', 'Coaster', 'Compass',
       'Convertible', 'Cool Ray', 'Cooper', 'Cooper Hatch', 'Cooper Paceman',
       'Cooper Roadster', 'Cordoba', 'Corolla', 'Corolla Cross', 'Corsa',
       'Corsa E', 'Country man', 'Countryman S', 'Coupe', 'Cressida', 'Creta',
       'Creta SU2', 'Cross', 'Crossland', 'Cruze', 'Cx 3', 'D max', 'DS5',
       'DS7', 'DX3', 'DX5', 'DX8S Coupe', 'Dart', 'Dashing', 'Discovery', 'Discovery sport', 'Doblo', 'Dokker', 'Dolphin', 'Ds4', 'Ds5', 'Ducato',
       'Durango', 'Duster', 'Dx7 prime', 'Dyna', 'E 180', 'E 200', 'E 200 AMG',
       'E 220', 'E 230', 'E 240', 'E 250', 'E 280', 'E 300', 'E 320', 'E 350',
       'E Golf', 'E-Tron', 'E-pace', 'EAGLE 580', 'EC 7', 'EC7', 'ENS1', 'EQB',
       'EX7', 'Eado', 'Eado plus', 'Eagle', 'Eagle pro', 'Echo', 'Eclipse',
       'Eclipse Cross', 'EcoSport', 'Elantra', 'Elantra AD', 'Elantra CN7',
       'Elantra Coupe', 'Elantra HD', 'Elantra MD', 'Emgrand 7', 'Emgrand X7',
       'Emkoo', 'Empow', 'Envy', 'Eos', 'Equinox', 'Ertiga', 'Escape', 'Escort', 'Ex 7', 'Excel',
       'F-150', 'F-Pace', 'F0', 'F3', 'F3R', 'FRV', 'FRV Cross', 'FSV',
       'Fabia', 'Familia', 'Family', 'Fantasia', 'Felicia', 'Fiesta', 'Florid',
       'Fluence', 'Focus', 'Formentor', 'Forte', 'Fortuner', 'Fortwo',
       'Frontera', 'Fusion', 'Fx', 'GLA', 'GLA 180', 'GLC 200',
       'GLC 200 AMG Imported', 'GLC 250', 'GLC 300', 'GLE', 'GLK', 'GLK 250',
       'GLK 300', 'GLK 350', 'GT', 'GX3 Pro', 'Galena', 'Galloper', 'Gen 2',
       'Gentra', 'Getz', 'Giulia', 'Giulietta', 'Glory', 'Glory 330', 'Glory Van', 'Golf', 'Golf 2',
       'Golf 3', 'Golf 4', 'Golf 5', 'Golf 6', 'Golf 7', 'Gran max',
       'Grand C4 Spacetourer', 'Grand Cerato', 'Grand Cherokee',
       'Grand Cherokee L', 'Grand Santa Fe', 'Grand i10', 'Grand terios',
       'Grand vitara', 'Grandis', 'Grandland', 'Granta', 'Gratour', 'Groove',
       'H1', 'H530', 'H6', 'H6 GT', 'HRV', 'HS', 'Hiace', 'Hilux', 'Hover',
       'Huge', 'I-pace', 'I10', 'I20', 'I30', 'ID 4', 'ID 6', 'IX 35', 'Ibiza',
       'Imperial', 'Impreza', 'Insignia', 'Ioniq', 'Ix20', 'J2', 'J3', 'J5', 'J7', 'JS3', 'JS4', 'Jetta', 'Jimny', 'John Cooper', 'Jolion',
       'Juke', 'Juliet', 'K01', 'K05S', 'K07S', 'K3', 'K5', 'K900', 'Kadjar',
       'Kamiq', 'Kancil', 'Karoq', 'Kelisa', 'Kodiaq', 'Komodo', 'Kuga', 'L3',
       'LR2', 'Lacetti', 'Lancer', 'Lancer Crystala', 'Lancer EX Shark',
       'Lancer Puma', 'Land Cruiser', 'Lanos', 'Lanos 2', 'Leganza', 'Leon',
       'Liberty', 'Linea', 'Lodgy', 'Logan', 'Lr4', 'M11', 'M12', 'M3', 'M50',
       'M60', 'MCV', 'MPV', 'MRV', 'MZ 40', 'Macan', 'Malibu', 'Maruti', 'Materia', 'Matiz', 'Matrix', 'Maxima',
       'Megane', 'Meriva', 'Microbus', 'Mini Cooper S', 'Minivan', 'Mirage',
       'Mk', 'Model 3', 'Model Y', 'Mohave', 'Mokka', 'Mondeo', 'Multivan',
       'N-Series', 'N300', 'Neon', 'New Star', 'Niva', 'Nubira 1', 'Nubira 2',
       'Octavia A4', 'Octavia A5', 'Octavia A7', 'Octavia A8', 'Okavango',
       'Oley', 'Opirus', 'Optima', 'Optra', 'Outlander', 'PT Cruiser',
       'PX Cargo', 'Pajero', 'Palio', 'Panda', 'Pandino', 'Parati',
       'Park Avenue', 'Passat', 'Pegas', 'Persona', 'Petra', 'Picanto', 'Pick up', 'Pickup', 'Polestar 2', 'Polo', 'Prado', 'Preve', 'Pride',
       'Punto', 'Punto evo', 'Q2', 'Q22', 'Q3', 'Q5', 'Q7', 'Qashqai',
       'RX5 Plus', 'Rainbow', 'Range Rover Evoque', 'Range Rover Sport',
       'Rapid', 'Rav 4', 'Renegade', 'Rio', 'Roomster', 'Rumion', 'Rush',
       'Rx5', 'S 280', 'S 300', 'S 320', 'S 350', 'S 500', 'S Presso', 'S2',
       'S3', 'S30', 'S4', 'S40', 'S5', 'S60', 'S80', 'SEL', 'SEL 280',
       'SIRION', 'Safary', 'Saga', 'Sandero', 'Sandero Step Way', 'Scala',
       'Scenic', 'Scirocco', 'Scorpio', 'Seltos', 'Sentra', 'Sephia', 'Shahin', 'Shuma',
       'Siena', 'Sienna', 'Sierra', 'Solaris', 'Sonata', 'Sonic', 'Sorento',
       'Soul', 'Spark', 'Spark Lite', 'Spectra', 'Splendor', 'Sportage',
       'Stanza', 'Stelvio', 'Sunny', 'Superb', 'Suran', 'Swift', 'Swift Dzire',
       'Sx4', 'Symbol', 'T-Series', 'T2', 'T33', 'T5 evo', 'T55', 'T600',
       'TXL', 'Tarraco', 'Tercel', 'Terios', 'Terrain', 'Thunderbird', 'Tiba',
       'Tiggo', 'Tiggo 3', 'Tiggo 4', 'Tiggo 7', 'Tiggo 7 pro',
       'Tiggo 7 pro max', 'Tiggo 8', 'Tiggo 8 Pro', 'Tiggo 8 Pro Max', 'Tigra', 'Tiguan', 'Tiida', 'Tipo', 'Tivoli',
       'Tivoli XLV', 'Toledo', 'Torres', 'Touareg', 'Town & Country',
       'Traverse', 'Trax', 'Trooper', 'Tucson', 'Tucson GDI',
       'Tucson Turbo GDI', 'Turan', 'V101', 'V3', 'V5', 'V6', 'Van', 'Vectra',
       'Veloster', 'Veloz', 'Verna', 'View', 'Vita', 'Vitara', 'Viva',
       'Voyager', 'Waja', 'Wira', 'Wrangler', 'Wrangler Unlimited',
       'X Pandino', 'X-Type', 'X1', 'X2', 'X25', 'X3', 'X3 M', 'X3 Pro', 'X35',
       'X4', 'X40', 'X5', 'X5 M', 'X6', 'X6 M', 'X7', 'X70', 'X70 Plus', 'X70S', 'X90', 'X90 Plus', 'X95', 'XA',
       'XC 40', 'XC60', 'XC90', 'XD', 'XF', 'XG', 'XJ', 'XTrail', 'XV',
       'Xceed', 'Xpander', 'Xpander Cross', 'Xplosion', 'Xsara', 'Yaris',
       'Yeti', 'Yukon', 'ZRV', 'ZS', 'ZX', 'lr3']





possible_make_models=sorted(df_cleaned['Make Model'].unique().tolist())


#print(possible_make_models)

#print(len(possible_make_models))




possible_colors=['Beige', 'Black', 'Blue', 'Bronze',
       'Brown', 'Champagne', 'Cyan', 'Dark blue', 'Dark green', 'Dark grey',
       'Dark red', 'Eggplant', 'Gold', 'Gray', 'Green', 'Mocha', 'Olive',
       'Orange', 'Petroleum', 'Purple', 'Red', 'Silver', 'White', 'Yellow']



possible_cities=['10th of Ramadan',
       '6 October', 'Abu Hummus', 'Abu Kabir', 'Abu Qir', 'Agamy', 'Ain Shams',
       'Al Shorouk', 'Alexandria', 'Amreya', 'Ashmoun', 'Aswan', 'Asyut',
       'Badr City', 'Banha', 'Beheira', 'Beni Suef', 'Bilbeis', 'Borg el arab',
       'Cairo', 'Dakahlia', 'Damanhur', 'Damietta', 'Dikirnis', 'Dokki', 'Dyarb Negm',
       'Edku', 'El Bagour', 'El Gamaleya', 'El Gouna', 'El Haram',
       'El Katameya', 'El Mahalla', 'El Manial', 'El Marg', 'El Minya',
       'El Qanater El Khayreya', 'El Salam City', 'El Senbellawein',
       'El Wadi El Gedid', 'El-Alamein', 'El-Arish', 'Faiyum', 'Faqous',
       'Gharbia', 'Giza', 'Heliopolis', 'Helwan', 'Hurghada', 'Imbaba',
       'Ismailia', 'Kafr El Zayat', 'Kafr Shukr', 'Kafr el-Dawwar',
       'Kafr el-Sheikh', 'Kerdasa', 'Khanka', 'Kom Hamada', 'Kom Ombo',
       'Luxor', 'Maadi', 'Madinaty', 'Mansoura', 'Marsa Alam', 'Marsa Matrouh',
       'Menouf', 'Minya Al Qamh', 'Mit Ghamr', 'Mohandessin', 'Mokattam', 'Monufia', 'Nasr city', 'New Administrative Capital', 'Nizwa',
       'Obour City', 'Port Said', 'Pyramids Gardens', 'Qalyub', 'Qalyubia',
       'Qena', 'Quesna', 'Ras Gharib', 'Red Sea', 'Rosetta', 'Sadat City',
       'Safaga', 'Saft El Laban', 'Sharm el-Sheikh', 'Sharqia',
       'Sheikh Zayed City', 'Sheraton', 'Shibin El Qanater', 'Shibin el Kom',
       'Shobra', 'Sohag', 'Suez', 'Tagamo3 - New Cairo', 'Tala', 'Tanta',
       'Touhk', 'Warraq', 'Zagazig', 'Zamalek', 'Zefta']







def one_hot_encode(x, possible_xs):
   
    # Check if the model is in the list of possible models
    if x not in possible_xs:
        raise ValueError(f"Model '{x}' is not in the list of possible xs.")
    
    # Create a list of zeros with the same length as the possible models
    encoded_x = [False] * len(possible_xs)
    
    # Find the index of the model in the list of possible models
    x_index = possible_xs.index(x)
    
    # Set the corresponding index in the encoded model list to 1
    encoded_x[x_index] = True
    
    df=pd.DataFrame(columns=possible_xs)
    df.loc[len(df)] = encoded_x

    
    return df




def label_encode_year(year):
    """
    Label encode the year based on custom mapping.
    
    Parameters:
    year (int): The year to encode.
    
    Returns:
    int: The encoded value.
    """
    # Check if the year is within the range 1970 to 2025
    if year < 1970 or year > 2025:
        raise ValueError("Year must be within the range 1970 to 2025.")
    
    # Label encode the year based on the custom mapping
    encoded_value = year - 1970
    
    return encoded_value


def label_encode_month(month):
    
    encoded_value = month - 4
    
    return encoded_value



def and_operation(boolean_list):
    for boolean_value in boolean_list:
        if not boolean_value:
            return False
    return True
    



# Input fields for user to enter car details
car_make_model = st.selectbox("Model", possible_make_models)
#car_model = st.text_input("Model")
year = st.number_input("Year", min_value=1980, max_value=2025, step=1)
mileage = float(st.number_input("Mileage", min_value=5000))
color = st.selectbox("Color", possible_colors)
city = st.selectbox("City", possible_cities)
automatic = st.radio("Automatic", ("Yes", "No"))
conditioner = st.radio("Has Air Conditioner", ("Yes", "No"))
power = st.radio("Has Power Steering", ("Yes", "No"))
remote = st.radio("Has Remote Control", ("Yes", "No"))
month = st.number_input("Month at which the ad was placed", min_value=4, max_value=6, step=1)



automatic = True if automatic == "Yes" else False
conditioner = True if conditioner == "Yes" else False
power = True if power == "Yes" else False
remote = True if remote == "Yes" else False



car_make_model

year=label_encode_year(year)


month=label_encode_month(month)

data = [
    [mileage, automatic, conditioner, power, remote]
]

# Assuming the first list contains the column names
column_names = ['Mileage', 'Automatic Transmission', 'Air Conditioner', 'Power Steering', 'Remote Control']

# Creating the DataFrame with column names as indexes
df = pd.DataFrame(data, columns=column_names)
df = pd.concat([df, one_hot_encode(color,possible_colors), one_hot_encode(car_make_model,possible_make_models),
                one_hot_encode(city,possible_cities)], axis=1)

data_temp = [
    [year, month]
]

# Assuming the first list contains the column names
column_names_temp = ['Model Year Encoded', 'Month Encoded']

# Creating the DataFrame with column names as indexes
df_temp = pd.DataFrame(data_temp, columns=column_names_temp)


df=pd.concat([df, df_temp], axis=1)

cairo_cities=['Obour City', 'Sheikh Zayed City', 'Nasr city', 'Heliopolis',
       'Shibin El Qanater', 'Dokki', 'Tagamo3 - New Cairo', 'Cairo',
       'Kafr el-Sheikh', 'Mansoura', 'Madinaty', 'Giza', 'Damietta',
       'El Haram', 'Helwan',
       'Pyramids Gardens', '6 October', 'Mohandessin',
       'Al Shorouk', 'Maadi', 'Sheraton',
       'Shobra',  'Badr City',
       '10th of Ramadan', 'Mokattam', 'El Manial',
       'El Katameya','Ain Shams','El Salam City', 'Qalyub', 'Zamalek', 
       'Amreya']


df['In Cairo'] = df.apply(lambda row: True if any(row[cairo_cities]) else False, axis=1)


#print(df.columns)

st.write(df)
df_scaled=scaler.transform(df)

st.write(df_scaled)

if st.button("Predict Price"):
    st.write("Price is ",int(model.predict(df_scaled)), " EGP")

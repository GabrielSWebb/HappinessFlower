import time
from datetime import datetime
from pathlib import Path
import pandas as pd

CHUNK_SIZE = 5000
DATASET_PATH = Path("Car_Parking_Sensor_Data_2019.csv")
DATETIME_FMT = "%m/%d/%Y %H:%M:%S %p"

df_train_list = [] 

AREAS = ["Kensington","Degraves","Drummond","Tavistock","Markilles","RACV","City Square","Chinatown","McKillop","Spencer","Windsor","Twin Towers","Rialto","West Melbourne","The Mac","University","Hyatt","Family","East Melbourne","Hardware","Courtney","Supreme","Mint","Library","County","Magistrates","Victoria Market","Banks","Regency","Titles","Princes Theatre","Jolimont","Southbank","Queensberry","Docklands"]

def adjust_time(row: pd.Series) -> None:
    dtm = datetime.strptime(row["ArrivalTime"], DATETIME_FMT)
    day_sec = dtm.hour * 3600 + dtm.minute * 60 + dtm.second
    day_15m = day_sec // 900
    row["ArrivalTime"] = "{} {:%A}".format(day_15m, dtm)
    dtm = datetime.strptime(row["DepartureTime"], DATETIME_FMT)
    day_sec = dtm.hour * 3600 + dtm.minute * 60 + dtm.second
    day_15m = day_sec // 900
    row["DepartureTime"] = "{} {:%A}".format(day_15m, dtm)
    return row

def process_dataframe1(df: pd.DataFrame) -> pd.DataFrame:

    """On enlève les raws avec des donnés < à 3 min dans la colonne DurationMinutes"""
    df = df[df["DurationMinutes"] >= 3].copy()

    """On enlève les raws avec des donnés NAN dans la colonne DurationMinutes"""
    df.dropna(subset = ['DurationMinutes'])

    """On enlève les raws avec des donnés NAN à la fois dans la colonne ArrivalTime et DepartureTime how=(all) """
    df = df.dropna(subset=['ArrivalTime', 'DepartureTime'], how='all')

    """On attribut un entier à chaque zone présentes dans la liste AREAS[]"""
    df["AreaName"] = df["AreaName"].replace(AREAS, range(len(AREAS)))

    """attribut int ou NA """
    df = df[pd.to_numeric(df['AreaName'], errors='coerce').notnull()]

    """On enlève les raws avec des donnés qui ne sont pas des entiers dans la colonne AreaName"""
    df['AreaName'] = df['AreaName'].astype(int)

    """float to int"""
    df['DurationMinutes'] = df['DurationMinutes'].astype(int)

    """On enlève les columns dont on ne va pas se servir"""
    df.drop(
        [
            "Sign",
            "StreetMarker",
            "SignPlateID",
            "Sign",
            "StreetId",
            "StreetName",
            "BetweenStreet1ID",
            "BetweenStreet1",
            "BetweenStreet2ID",
            "BetweenStreet2",
            "SideOfStreet",
            "SideOfStreetCode",
            "SideName", 
            "BayId",
            "InViolation",],axis=1,inplace=True)
    df = df.transform(adjust_time, axis=1)

    return df
        
def load_melbourne() -> None:
    """Load the Melbourne data."""
    df: pd.DataFrame
    for df in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE):
        """Data processing"""
        df = process_dataframe1(df)
        df_train_list.append(df)
        
load_melbourne()
"""Export DataFrame avec des données exploitables"""
result = pd.concat(df_train_list)
result.to_csv('clean_data.csv', index=False)

df1 = pd.read_csv('clean_data.csv')
"""Export DataFrame avec le nombre de capteurs par Areas"""
df1.groupby(by='AreaName', as_index=False).agg({'DeviceId': pd.Series.nunique}).to_csv('clean_data_area_nb_device.csv', index=False)
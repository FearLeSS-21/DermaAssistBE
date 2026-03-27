import mysql.connector 
from dotenv import load_dotenv 
import os 

load_dotenv()
try :
    connection =mysql.connector.connect(
    host =os.getenv("MYSQL_HOST","localhost"),
    user =os.getenv("MYSQL_USER"),
    password =os.getenv("MYSQL_PASSWORD"),
    database =os.getenv("MYSQL_DATABASE")
    )
    print("Connection successful!")
    connection.close()
except mysql.connector.Error as err :
    print(f"Error: {err}")


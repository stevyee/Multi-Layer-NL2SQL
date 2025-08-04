from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
class DatabaseConnection:
    def __init__(self):
        self.tunnel = None
        self.engine = None
        self.db = None
    
    def connect(self, connection_settings):
        """
        Establishes SSH tunnel and database connection using provided settings
        """
        try:
            # Close existing connections if any
            # self.cleanup()
            
            # Create SSH tunnel
            self.tunnel = SSHTunnelForwarder(
                (connection_settings["ssh_address"], connection_settings["ssh_port"]),
                ssh_username=connection_settings["ssh_username"],
                ssh_pkey=connection_settings["ssh_key_path"],
                remote_bind_address=(connection_settings["remote_bind_address"], 
                                   connection_settings["remote_bind_port"])
            )
            self.tunnel.start()
            
            # Create database engine
            self.engine = create_engine(
                f'mysql+pymysql://{connection_settings["db_user"]}:'
                f'{connection_settings["db_password"]}@127.0.0.1:'
                f'{self.tunnel.local_bind_port}/{connection_settings["db_name"]}'
            )
            # Create SQLDatabase instance
            self.db = SQLDatabase(self.engine)
            
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            # self.cleanup()
            return False
    
    # def cleanup(self):
    #     """
    #     Closes all connections
    #     """
    #     if self.engine:
    #         self.engine.dispose()
    #         self.engine = None
    #     if self.tunnel:
    #         self.tunnel.stop()
    #         self.tunnel = None
    #     self.db = None
    
    def get_engine(self):
        return self.engine
    
    def get_db(self):
        return self.db

# Create a single instance to be used across the application
db_connection = DatabaseConnection()
# For backwards compatibility
def get_engine():
    return db_connection.get_engine()



# Define connection_settings with your actual values


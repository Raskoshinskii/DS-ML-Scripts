from tqdm.notebook import tqdm as log_progress

# Progress bar
for i in log_progress(range(1)):
    print(i)    
   
# Config reading 
def read_config(file_name):
    # Required Libs:
    import os, configparser
    from pathlib import Path
    
    dirname = os.path.dirname(__file__)
    config_path = os.path.abspath(Path(dirname).parent)
    config_path = os.path.join(config_path, file_name)
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# DB Interface (can be applied as a template for any DB)
class DataBaseHandler:
     def __init__(self, login: str, pwd: str):
            self.login = login
            self.pwd = pwd
            
    def db_connector(func):
        def with_connection(self, *args, **kwargs):
            connection = cs_Oracle.connect(self.login, self.pwd)
            try:
                result = func(self, connection, *args, **kwargs)
            except Exception as e:
                print(e)
                connection.rollback()
                raise
            else:
                connection.commit()
            finally:
                connection.close()
            return result
        return with_connection
    
    def make_select(self, connection, query):
        cursor = connection.cursor()
        cursor.execute()
        return cursor.fetchall()
    
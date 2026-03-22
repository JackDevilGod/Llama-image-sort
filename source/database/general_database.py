import sqlite3

from pathlib import Path


class ImageDataBase:
    def __init__(self) -> None:
        self.database_path: Path = Path(__file__).parent.joinpath(
                                        "resource").joinpath(
                                        "imagedatabase.db")
        
        self.database_connection = sqlite3.connect(self.database_path)
        self.cursor = self.database_connection.cursor()
        
        if self.cursor.execute("SELECT name FROM sqlite_master WHERE name='image'") is None:
            self.cursor.execute("CREATE  TABLE image(path, hash, tags)")
        

if __name__ == "__main__":
    test = ImageDataBase()
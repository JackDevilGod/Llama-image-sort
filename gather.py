from pathlib import Path
from source.database.general_database import ImageDatabaseManager


def main():
    database_path = ""
    out_directory = ""
    tags = []
    
    database = ImageDatabaseManager(Path(database_path))
    
    image_out: list[Path] = database.get_images(tags)
    out_path = Path(out_directory)
    
    for p in image_out:
        p.move_into(out_path)


if __name__ == "__main__":
    main()

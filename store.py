from pathlib import Path

from source.taggers.ollama_tagger import Otagger
from source.taggers.wd_tagger import WDTagger
from source.database.general_database import ImageDatabaseManager


def main():
    tagger_togle = 0
    directory_path = ""
    database_path = ""
    
    if tagger_togle == 1:
        tagger = Otagger("qwen3.5:9b")
    else:
        tagger = WDTagger("SmilingWolf/wd-vit-tagger-v3")
        
    database = ImageDatabaseManager(Path(database_path))
    
    queue = [Path(directory_path)]
    
    while queue:
        current = queue.pop(0)
        
        if current.is_dir():
            queue += [_ for _ in current.iterdir()]
            continue
        
        tags = tagger.get_tags(current)
        
        database.add_image(current, tags=tags)
    
    

if __name__ == "__main__":
    main()
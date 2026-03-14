import ollama
from pathlib import Path


class Otagger:
    def __init__(self,model_name: str) -> None:
        self._model_name: str = model_name
        ollama.pull(self._model_name)
        self._default_prompt: Path = Path(__file__).parent.joinpath(
                                     "resources").joinpath(
                                     "prompt.txt")
        
    def get_tags(self, image_path: Path) -> list[str]:
        prompt = self._default_prompt.read_text()

        response = ollama.chat(
            model=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)]
                }
            ]
        )

        text = response["message"]["content"].strip()

        tags = [t.strip() for t in text.split(",") if t.strip()]
        return tags


if __name__ == "__main__":
    tagger = Otagger("qwen3.5:9b")
    
    tags = tagger.get_tags(Path(r"D:\Pictures\vlcsnap-2026-02-07-22h47m08s252.png"))
    print(tags)

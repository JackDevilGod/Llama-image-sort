import ollama




def main():
    model: str = "qwen3.5:9b"

    pull_repsonse: ollama.ProgressRepsonse= ollama.pull(model)

    if pull_repsonse.status != "success":
        print("Failed to get Model, please install ollama or start it.")
        exit(1)

    print(f"{model} Found.")


if __name__ == "__main__":
    main()
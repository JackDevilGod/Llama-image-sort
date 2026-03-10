import ollama


MODEL: str = "qwen3.5:2b"


def main():
    pull_repsonse: ollama.ProgressRepsonse= ollama.pull(MODEL)

    if pull_repsonse.status != "success":
        print("Failed to get Model, please install ollama or start it.")
        exit(1)

    print(f"{MODEL} Found.")


if __name__ == "__main__":
    main()
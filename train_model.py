from src.pipeline import train_rul_model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the battery RUL model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="rul_model.pkl")
    args = parser.parse_args()
    print(train_rul_model(args.data, args.output))

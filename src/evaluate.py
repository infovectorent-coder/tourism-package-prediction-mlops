from train import train

MIN_ACCURACY = 0.80
MIN_F1 = 0.75

if __name__ == "__main__":
    acc, f1 = train()
    if acc < MIN_ACCURACY or f1 < MIN_F1:
        raise SystemExit(f"Model failed thresholds: acc={acc}, f1={f1}")
    print("Model passed thresholds")

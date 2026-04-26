"""
CLI Disease Predictor — Run directly from terminal
Usage: python cli.py
"""

from model import load_model, predict_disease, SYMPTOMS


def print_banner():
    print("\n" + "═" * 55)
    print("   🧬  MedScan — AI Disease Prediction System")
    print("═" * 55)


def print_results(results):
    medals = ["🥇", "🥈", "🥉"]
    print("\n📋  TOP PREDICTIONS")
    print("─" * 55)
    for i, r in enumerate(results):
        medal = medals[i] if i < len(medals) else f"#{i+1}"
        bar_len = int(r["confidence"] / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)

        print(f"\n{medal}  {r['disease']}  [{r['severity']}]")
        print(f"   Confidence: {r['confidence']:.1f}%")
        print(f"   [{bar}]")
        print(f"   {r['description']}")

        print("\n   🛡️  Precautions:")
        for p in r["precautions"]:
            print(f"      • {p}")

        print("\n   💊  Medicines:")
        for m in r["medicines"]:
            print(f"      • {m}")

        print("\n" + "─" * 55)


def main():
    print_banner()
    print("\nLoading AI model...")
    model = load_model()
    print("✅ Model ready!\n")

    print("Available symptoms (type numbers separated by commas):\n")
    for i, s in enumerate(SYMPTOMS):
        label = s.replace("_", " ").title()
        print(f"  {i+1:2d}. {label}", end="\t" if (i + 1) % 3 != 0 else "\n")
    print("\n")

    while True:
        raw = input("Enter symptom numbers (e.g. 1,3,5) or 'q' to quit: ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("\n👋 Goodbye! Stay healthy.\n")
            break

        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            selected = [SYMPTOMS[i] for i in indices if 0 <= i < len(SYMPTOMS)]
        except (ValueError, IndexError):
            print("❌ Invalid input. Enter comma-separated numbers.\n")
            continue

        if not selected:
            print("❌ No valid symptoms selected.\n")
            continue

        print(f"\n🔬 Analyzing {len(selected)} symptom(s): {', '.join(s.replace('_',' ') for s in selected)}\n")
        results = predict_disease(selected, model)
        print_results(results)


if __name__ == "__main__":
    main()

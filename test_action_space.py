import re, random, yaml, torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_cfg(path="config.yaml"):
    with open(path) as f:
        return OmegaConf.create(yaml.safe_load(f))


def build_prompt(floor: float, budget: float, ask: float) -> str:
    example_dollar = floor + 0.25 * (budget - floor)
    return (
        f"Floor: {floor:.0f}\n"
        f"Budget: {budget:.0f}\n"
        f"Seller Ask: {ask:.0f}\n"
        f"Negotiation Range: {floor:.0f} to {budget:.0f}\n"
        "Respond with a single decimal number between 0.0 and 1.0. "
        "No words, no percent sign, no explanation. Only the number.\n"
        "Example output: 0.25\n"
        "Output:"
    )


def extract_float(text: str) -> float | None:
    # Match standalone floats in [0, 1]: 0, 0.xx, 1, 1.0
    matches = re.findall(r'(?<!\d)(0(?:\.\d+)?|1(?:\.0*)?)(?!\d)', text)
    for m in matches:
        v = float(m)
        if 0.0 <= v <= 1.0:
            return v
    return None


def main():
    cfg = load_cfg()
    ckpt = cfg.paths.checkpoints.grpo_diplomacy
    dtype = torch.bfloat16 if cfg.grpo_diplomacy.bf16 else torch.float32

    print(f"Loading model from {ckpt} ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype, device_map="auto")
    model.eval()
    print("Model loaded.\n")

    rng = random.Random(42)
    n = 20
    valid_count = 0
    in_range_count = 0
    raw_examples = []

    for i in range(n):
        floor  = round(rng.uniform(50, 300))
        budget = round(rng.uniform(400, 800))
        ask    = round(budget + rng.uniform(50, 300))  # always above budget

        prompt = build_prompt(floor, budget, ask)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["\n"],
                tokenizer=tokenizer,
            )
        raw = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        pct      = extract_float(raw)
        valid    = pct is not None
        dollar   = floor + pct * (budget - floor) if valid else None
        in_range = valid and (floor <= dollar <= budget)

        if valid:    valid_count += 1
        if in_range: in_range_count += 1
        if len(raw_examples) < 5:
            raw_examples.append(raw)

        print(f"[{i+1:02d}] floor=${floor}  budget=${budget}  ask=${ask}")
        print(f"      raw     : {repr(raw[:100])}")
        print(f"      pct     : {pct}  =>  dollar: {'${:.2f}'.format(dollar) if dollar is not None else 'N/A'}"
              f"  in_range: {in_range}\n")

    print("=" * 55)
    print(f"Valid float extracted : {valid_count}/{n}  ({100*valid_count/n:.1f}%)")
    print(f"In-range dollar amount: {in_range_count}/{n}  ({100*in_range_count/n:.1f}%)")
    if valid_count:
        print(f"Of valid, in-range    : {in_range_count}/{valid_count}  ({100*in_range_count/valid_count:.1f}%)")
    print("\nRaw output examples (first 5):")
    for ex in raw_examples:
        print(f"  {repr(ex)}")
    print("=" * 55)


if __name__ == "__main__":
    main()

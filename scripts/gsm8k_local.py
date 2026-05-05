from datasets import load_dataset

out_dir = "/scratch/gpfs/MENGDIW/jg8305/.cache/huggingface/gsm8k_local"

ds = load_dataset("openai/gsm8k", "main")
ds.save_to_disk(out_dir)

print(f"Saved GSM8K to {out_dir}")
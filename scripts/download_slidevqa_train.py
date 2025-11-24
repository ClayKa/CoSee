from datasets import load_dataset

def main():
    ds_train = load_dataset(
        "NTT-hil-insight/SlideVQA",
        split="train",
        cache_dir="data/hf_cache/slidevqa",
    )
    print(ds_train)
    print("Num examples:", len(ds_train))
    print("Columns:", ds_train.column_names)

if __name__ == "__main__":
    main()

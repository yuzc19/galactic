import datasets
from datasets import Dataset
from galactic import GalacticDataset
from litdata import optimize
from litdata.streaming import StreamingDataset, TokensLoader
from transformers import AutoTokenizer


def transform_dataset(path):
    dataset = StreamingDataset(
        input_dir=path,
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    dataset = Dataset.from_list([{"input_ids": d} for d in dataset])
    print(len(dataset[0]["input_ids"]))

    # Load pythia tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")

    def preprocess_data(examples):
        text = pythia_tokenizer.batch_decode(
            examples["input_ids"],
            skip_special_tokens=True,
        )
        return {"text": text}

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=1024,
        num_proc=8,
        remove_columns=dataset.column_names,
    )
    total_samples = len(dataset)
    print("Total number of examples:", total_samples)

    dataset.to_json(path + ".jsonl")


def plot(path):
    path = "/data/users/zichunyu/data/fineweb/sample-350BT/val.jsonl"
    dataset.to_json(path)

    dataset = GalacticDataset.from_jsonl(path)
    total_samples = len(dataset)

    # 350M to 11000 clusters
    dataset.get_embeddings(input_field="text", backend="cpu")
    dataset.cluster(n_clusters=total_samples // 250)
    dataset.get_cluster_info(context_fields=["text"])
    dataset.save("my_dataset-350BT.jsonl", overwrite=True)

    dataset = GalacticDataset.from_jsonl("my_dataset-350BT.jsonl")
    dataset.cluster(n_clusters=5, overwrite=True)
    dataset.get_cluster_info(context_fields=["text"])
    dataset.reduce_embedding_dim(
        n_dims=2,
        method="umap",
        new_column="__embedding_umap_2",
        embedding_field="__embedding",
    )
    dataset.plot_embeddings(
        embedding_field="__embedding_umap_2",
        color_by="__cluster",
        width=16,
        height=12,
        save_path="umap.png",
        dot_size=20,
    )


def concatenate_datasets():
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"/data/users/zichunyu/data/fineweb/sample-350BT/train/0_embedding/{i}"
            )
            for i in range(8)
        ]
    )
    dataset.to_json(
        "/data/users/zichunyu/data/fineweb/sample-350BT/train/0_embedding.jsonl"
    )


def find_nearest_neighbors():
    path = "/data/users/zichunyu/data/fineweb/sample-350BT/val_embedding.jsonl"
    dataset = GalacticDataset.from_jsonl(path)
    print(len(dataset))
    rank = 8
    neighbors = dataset.get_nearest_neighbors(dataset[-rank]["__embedding"], 1000)
    # print(neighbors)

    # total_samples = len(dataset)
    # dataset.cluster(n_clusters=total_samples // 20000)
    # dataset.get_cluster_info(context_fields=["text"])
    # dataset.semdedup(target_retention=0.7)

    indices = list(reversed([n["__id"] for n in neighbors]))
    indices.append(dataset[-rank]["__id"])
    # indices = [d["__id"] for d in dataset]
    print(len(indices))
    dataset = StreamingDataset(
        input_dir="/data/users/zichunyu/data/fineweb/sample-350BT/val",
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    optimize(
        fn=lambda index: dataset[index],
        inputs=indices,
        output_dir=f"/data/users/zichunyu/lit-gpt/data/neighbors-{rank-1}",
        num_workers=8,
        chunk_bytes="200MB",
    )


# concatenate_datasets()

# transform_dataset("/data/users/zichunyu/data/fineweb/sample-350BT/train/0")

# find_nearest_neighbors()

# exit(0)

# BUG
# dataset = GalacticDataset.from_hugging_face_stream(
#     "/data/users/zichunyu/data/fineweb/sample-350BT/val-hf",
#     # "HuggingFaceFW/fineweb",
#     # config_name="sample-10BT",
#     cache_dir="/data/users/zichunyu/data/hf_cache",
#     split="train",
#     # filters=[lambda x: len(x["text"]) < 1024],
#     dedup_fields=["text"],
#     # max_samples=total_samples,
# )

# path = "/data/users/zichunyu/data/fineweb/sample-350BT/val.jsonl"
path = "/data/users/zichunyu/data/fineweb/sample-350BT/train/0_embedding.jsonl"
# dataset = GalacticDataset.from_jsonl(path)
dataset = GalacticDataset.from_hugging_face(path, "train")
# add a new column to the dataset (int id)
# dataset = dataset.map(lambda _, i: {"id": i}, with_indices=True)
total_samples = len(dataset)
print("Total number of examples:", total_samples)

# DsDm: 11,000 clusters, deduplicating down to 20% of the original dataset, 42554392 (216948746)
# dataset.get_embeddings(input_field="text", backend="gpu")
dataset.cluster(n_clusters=total_samples // 20000)
# dataset.get_cluster_info(context_fields=["text"])
dataset.semdedup(target_retention=0.7)
dataset.drop_column("text")
dataset.save(
    "/data/users/zichunyu/data/fineweb/sample-350BT/train/0_semdedup.jsonl",
    overwrite=True,
)

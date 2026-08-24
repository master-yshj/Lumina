import json
import pathlib

PROCESS_ROOT = pathlib.Path(__file__).parent
train_path = PROCESS_ROOT / 'cmrc2018_train.json'
valid_path = PROCESS_ROOT / 'cmrc2018_valid.json'
test_path = PROCESS_ROOT / 'cmrc2018_test.json'
tiny_target_dir = PROCESS_ROOT / 'tiny_cmrc2018'
base_target_dir = PROCESS_ROOT / 'base_cmrc2018'
full_target_dir = PROCESS_ROOT / 'full_cmrc2018'

dataset_size = {
    "tiny": {
        "train": 1000,
        "valid": 300,
        "test": 100,
    },
    "base": {
        "train": 8000,
        "valid": 2000,
        "test": 1000,
    },
    "full": {
        "train": None,
        "valid": None,
        "test": None,
    }
}

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def expand_dict(data_dict):
    data_list = []
    data = data_dict["data"]
    for paragraphs_dict in data:
        paragraphs = paragraphs_dict["paragraphs"]
        for paragraph in paragraphs:
            paragraph_id = paragraph["id"]
            context = paragraph["context"]
            qas = paragraph["qas"]
            for qa in qas:
                question = qa["question"]
                qa_id = qa["id"]
                answers = qa["answers"]
                data_dict = {
                    "id": qa_id,
                    "context": context,
                    "question": question,
                    "answers": answers,
                }
                data_list.append(data_dict)
    return data_list

def truncation_with_limit(data_list, limit):
    limit_data_list = data_list[:limit]
    return limit_data_list

def make_target_dir(dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)

def write_jsonl(target_path, limit_data_list):
    with open(target_path, 'w', encoding='utf-8') as f:
        for data in limit_data_list:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

def generate(size_name, raw_train_file, raw_valid_file, raw_test_file, target_dir):
    raw_train_dict = load_json(raw_train_file)
    raw_valid_dict = load_json(raw_valid_file)
    raw_test_dict = load_json(raw_test_file)
    train_data_list = expand_dict(raw_train_dict)
    valid_data_list = expand_dict(raw_valid_dict)
    test_data_list = expand_dict(raw_test_dict)
    train_size = dataset_size[size_name]["train"]
    valid_size = dataset_size[size_name]["valid"]
    test_size = dataset_size[size_name]["test"]
    train_data_limit_list = truncation_with_limit(train_data_list, train_size)
    valid_data_limit_list = truncation_with_limit(valid_data_list, valid_size)
    test_data_limit_list = truncation_with_limit(test_data_list, test_size)
    make_target_dir(target_dir)
    train_target_path = target_dir / f"{size_name}_cmrc2018_train.jsonl"
    valid_target_path = target_dir / f"{size_name}_cmrc2018_valid.jsonl"
    test_target_path = target_dir / f"{size_name}_cmrc2018_test.jsonl"
    write_jsonl(train_target_path, train_data_limit_list)
    write_jsonl(valid_target_path, valid_data_limit_list)
    write_jsonl(test_target_path, test_data_limit_list)

def main():
    generate("tiny", train_path, valid_path, test_path, tiny_target_dir)
    generate("base", train_path, valid_path, test_path, base_target_dir)
    generate("full", train_path, valid_path, test_path, full_target_dir)

if __name__ == "__main__":
    main()
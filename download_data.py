import os
from datasets import load_dataset
from datasets import disable_caching

#os.environ["HF_HOME"] = "./data"
#data = load_dataset("HuggingFaceFW/fineweb-2", name="kor_Hang", cache_dir = './data')
# datasets 전용 캐시 위치
os.environ["HF_DATASETS_CACHE"] = "/nfsdata/languageAI/users/ocw/data/datasets_cache"

# (필요하다면) transformers 모델·토크나이저 캐시까지 한꺼번에 바꾸고 싶다면
os.environ["TRANSFORMERS_CACHE"] = "/nfsdata/languageAI/users/ocw/data/transformers_cache"
data = load_dataset("HuggingFaceFW/fineweb", "sample-350BT", cache_dir = '/nfsdata/languageAI/users/ocw/data/datasets_cache')
data.save_to_disk('/nfsdata/languageAI/users/ocw/data/fineweb-350BT')
# data = load_dataset("HuggingFaceFW/fineweb", name="sample-99BT", cache_dir = './data')

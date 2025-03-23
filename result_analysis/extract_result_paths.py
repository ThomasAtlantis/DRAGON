"""
Usage example:
python result_analysis/extract_result_paths.py \
    --input_file logs/lm_qwen_wikitext2_seq512.log \
    --output_file result_analysis/result_paths_lm_wikitext2_qwen_seq512.json
"""
import sys; sys.path.append('.')
import re, json
from dragon.utils.configure import Configure, Field

class Config(Configure):
    input_file = Field(str, required=True, help="Input file path")
    output_file = Field(str, required=True, help="Output file path")
config = Config()
config.parse_sys_args()

result_paths = {}
with open(config.input_file, 'r') as log:
    for line in log:
        title = re.match(r"Evaluating ([^ ]+)(?: with n_docs=(\d+))?\n", line)
        paths = re.search(r"`([^`]+)`", line)
        if title:
            method, n_docs = title.group(1), title.group(2)
            method = method.replace("NoRetrieval", "W/O Retrieval")
            method = method.replace("cloud", "Cloud")
            method = method.replace("device", "Device")
            n_docs = int(n_docs) if n_docs else 0
        elif paths:
            result_paths.setdefault(f"{n_docs=}", {}).update({
                method: paths.group(1)
            })
with open(config.output_file, 'w') as out:
    json.dump(result_paths, out, indent=4)

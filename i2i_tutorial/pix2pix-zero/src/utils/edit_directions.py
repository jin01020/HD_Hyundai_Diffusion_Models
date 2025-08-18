import os
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


"""
This function takes in a task name and returns the direction in the embedding space that transforms class A to class B for the given task.

Parameters:
task_name (str): name of the task for which direction is to be constructed.

Returns:
torch.Tensor: A tensor representing the direction in the embedding space that transforms class A to class B.

Examples:
>>> construct_direction("cat2dog")
"""

def construct_direction(task_name):
    
    # src2dst 에서 src 와 dst 분리 
    (src, dst) = task_name.split("2")
    # 미리 저장된 embedding 경로
    emb_dir = f"assets/embeddings_sd_1.4"
    # embs_a, emb_b에 각각 해당되는 embedding을 불러옴. 
    embs_a = torch.load(os.path.join(emb_dir, f"{src}.pt"), map_location=device)
    embs_b = # FILL 

    # 두 embedding의 차이를 이용하여 direction을 뽑아냄.
    direction = # FILL # dimension 꼭 맞춰줄 것!
    return direction

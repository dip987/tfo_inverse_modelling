import time
from tqdm import tqdm

print("Start")
with tqdm(total=20) as pbar:
    for i in range(20):
        time.sleep(1)
        pbar.update(1)


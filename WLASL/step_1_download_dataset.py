import fiftyone as fo
import fiftyone.utils.huggingface as fouh


dataset = fouh.load_from_hub("Voxel51/WLASL", name="wlasl", max_samples=100)
dataset = fo.load_dataset("wlasl")
print(dataset)

session = fo.launch_app(dataset)

# This line is important to keep the session alive
session.wait()

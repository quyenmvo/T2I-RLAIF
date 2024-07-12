import torch-fidelity
from diffusers import StableDiffusionPipeline
from ml_collections import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config.py", "Validating configuration.")

with open("valid_dataset.json", "r") as f:
    dataset = json.load(f)

class TransformPILtoRGBTensor:
    def __call__(self, img):
        vassert(type(img) is Image.Image, "Input is not a PIL.Image")
        img = img.resize((512,512))
        return F.pil_to_tensor(img)


class ImagesDataset(Dataset):
    def __init__(self, images_list, transforms=None):
        self.images_list = images_list
        self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, i):
        path = self.images_list[i]
        img = self.transforms(img)
        return img

class ImagesPathDataset(Dataset):
    def __init__(self, path_list, transforms=None):
        self.path_list = path_list
        self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, i):
        path = self.path_list[i]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img
config = FLAGS.config
def evaluate(model_path, lora_path, dataset):
    lora_layer = ""
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline.unet.load_attn_procs(lora_path)
    images_list = []
    for prompt in dataset:
        image = pipeline(prompt, guidance_scale = 5.0)[0]
        images_list.append(image)
    generator = ImagesDataset(images_list)
    path_list = os.listdir("./ground_truth")
    generator0 = ImagesDataset(path_list)
    metrics_dict = torch_fidelity.calculate_metrics(
                input1=generator,
                input2=generator0,
                cuda=True,
                isc=True,
                fid=True,
                verbose=False,
            )
    is_score = metrics_dict['inception_score_mean']
    fid_score = metrics_dict['frechet_inception_distance']
    return is_score, fid_score
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import time

class VLM:
    def __init__(self, vlm_name: str, image):
        
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        
        print(f"VLM {vlm_name} runs on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        self.model = AutoModel.from_pretrained(
            vlm_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(vlm_name, trust_remote_code=True)

        self.image = image
        self.image_size = 448 # image will be resized to (image_size x image_size) for fast processing
        
        # set the max number of tiles in `max_num`
        self.pixel_values = self.load_image(max_num=6).to(torch.float16).cuda()

        # Test param modifications
        self.generation_config = dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.7,
            repetition_penalty=1.1,
            )

        self.prompt = "List the objects, with only one object per line"

    def run(self):
        # single-round single-image conversation
        start = time.time()
        response = self.model.chat(self.tokenizer, self.pixel_values, self.prompt, self.generation_config)
        end = time.time()
        print(f"VLM Inference time = {end - start}s")
        print(f"VLM Prompt = {self.prompt}")
        print(f"VLM Response = {response}")
        return response


    def build_transform(self):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform


    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.image_size * self.image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


    def dynamic_preprocess(self, min_num=1, max_num=6, use_thumbnail=False):
        orig_width, orig_height = self.image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height)

        # calculate the target width and height
        target_width = self.image_size * target_aspect_ratio[0]
        target_height = self.image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = self.image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self.image_size)) * self.image_size,
                (i // (target_width // self.image_size)) * self.image_size,
                ((i % (target_width // self.image_size)) + 1) * self.image_size,
                ((i // (target_width // self.image_size)) + 1) * self.image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = self.image.resize((self.image_size, self.image_size))
            processed_images.append(thumbnail_img)
        return processed_images


    def load_image(self, max_num=6):
        transform = self.build_transform()
        images = self.dynamic_preprocess(use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values


from src.vlm import VLM

# import torch
# from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import base64
from PIL import Image
from lang_sam import LangSAM
from io import BytesIO
import skimage.measure as sim
import skimage.transform as sit
import textwrap


# def save_image(image_path: str, image_data):
#     cv.imwrite(image_path, image_data)


class Perception:
    def __init__(self, vlm_name: str, vlm_provider: str):
        self.vlm_name = vlm_name
        self.vlm_provider = vlm_provider
        if vlm_provider != "Ollama":
            raise NotImplementedError(
                f"VLM provider {vlm_provider} not implemented yet, only Ollama is supported."
            )
        self.seg_model_name = "language-segment-anything"

        self.environment_description_list = []  # ["figurine", "cup", "table"]
        self.centers_location = []  # [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
        self.environment_pos = (
            {}
        )  # {'figurine':[x1,y1,z1], 'cup':[x2,y2,z2], 'table':[x3,y3,z3]}
        self.mask_opacity = 0.2

        self.image = None
        self.frame_with_masks_and_centers = None

    def build_frame_with_masks_and_centers(self, original_frame, masks, centers, labels):
        """
        Combine original frame + yellow semi-transparent masks + red centroids with labels.
        """
        # Ensure frame is BGR uint8
        frame = original_frame.copy()
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame.shape[2] == 3:
            # If this was in RGB (from PIL), convert to BGR
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # Create an overlay for mask blending
        overlay = frame.copy()
        mask_color = (0, 255, 255)  # yellow in BGR

        for mask in masks:
            if mask is None:
                continue

            # Ensure mask is binary uint8
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            mask_bin = (mask > 127).astype(np.uint8)

            # Colorize the mask and overlay it
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[:, :] = mask_color
            overlay = np.where(mask_bin[..., None].astype(bool), colored_mask, overlay)

        # Blend the overlay and the original frame
        blended = cv.addWeighted(frame, 1 - self.mask_opacity, overlay, self.mask_opacity, 0)

        # Draw centroids and labels
        for (x, y, _), label in zip(centers, labels):
            x, y = int(x), int(y)
            cv.circle(blended, (x, y), 6, (0, 0, 255), -1)  # red dot
            cv.putText(
                blended,
                label,
                (x + 10, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

        # Save result (keep as RGB for consistency with self.image)
        self.frame_with_masks_and_centers = blended


    def centroid_segmentation(self, map):
        """
        Connected component analysis to find the region with the highest median value, and return the centroid of each region.
        """
        resize_image_format = (64, 64)
        im_resized = sit.resize(map, resize_image_format)
        # Apply gaussian blur, threshold and finally labeling the image
        label_map = sim.label(cv.GaussianBlur(im_resized, ksize=(5, 5), sigmaX=1) > 0.3)

        # segmentation
        regions = sim.regionprops(label_map)

        # If no regions are found, return None
        if regions == []:
            return None, None

        # find the region with the highest median value
        median, best_region = -1, None
        for region in regions:
            values = im_resized[region.coords[:, 0], region.coords[:, 1]]
            if np.median(values) > median:
                median = np.median(values)
                best_region = region

        # Calculate the centroid and bounding box of the best region
        centroid = list(best_region.centroid)
        centroid[0] *= map.shape[0] / resize_image_format[0]
        centroid[1] *= map.shape[1] / resize_image_format[1]

        bbox = list(best_region.bbox)
        bbox[0] *= map.shape[0] / resize_image_format[0]
        bbox[2] *= map.shape[0] / resize_image_format[0]
        bbox[1] *= map.shape[1] / resize_image_format[1]
        bbox[3] *= map.shape[1] / resize_image_format[1]

        centroid = centroid[::-1]
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

        return centroid, bbox

    def segmentation(self):
        print(f"Run Image Segmentation model {self.seg_model_name}")
        # Initialize model
        model = LangSAM()

        # Convert PIL to RGB and keep original NumPy for processing
        image_pil = self.image.convert("RGB")
        image_np = np.array(image_pil)

        imgs_seg = []
        centers  = []
        not_found = []

        # Loop over each prompt individually
        for i, prompt in enumerate(self.environment_description_list):
            result = model.predict([image_pil], [prompt])[0]

            # Take only the highest‑confidence mask
            masks  = result["masks"]
            scores = result["scores"]
            if scores is None or np.size(scores) == 0 or masks is None or len(masks) == 0:
                print(f"No segmentation found for '{prompt}', skipping.")
                not_found.append(prompt)
                continue

            best_idx = int(np.argmax(scores))
            best_mask = (masks[best_idx].astype(np.uint8) * 255)

            # Compute centroid & bbox
            center, bbox = self.centroid_segmentation(best_mask)
            if center is None or bbox is None:
                not_found.append(prompt)
                continue

            # Store 2D center + dummy Z=0
            centers.append([center[0], center[1], 0])

            # Draw box+center on mask for visualization
            vis = best_mask.copy()
            x0, y0, x1, y1 = map(int, bbox)
            cv.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 3)
            cv.circle(vis, (int(center[0]), int(center[1])), 5, (255,255,255), -1)
            imgs_seg.append(vis)

        # Remove not‑found prompts from your list
        for nf in not_found:
            print(f"Object: {nf} not found, removing from descriptions")
            self.environment_description_list.remove(nf)

        # Rescale centers back to original image dimensions
        orig_w, orig_h = self.image.size
        mask_h, mask_w = (imgs_seg[0].shape[:2] if imgs_seg else (1,1))
        for c in centers:
            c[0] = c[0] * orig_w / mask_w
            c[1] = c[1] * orig_h / mask_h

        # Draw final centers on a copy of the original
        final_img = image_np.copy()
        for x, y, _ in centers:
            cv.circle(final_img, (int(x), int(y)), 20, (255, 0, 0), -1)

        # Build composite frame with masks and centers
        self.build_frame_with_masks_and_centers(image_np, imgs_seg, centers, self.environment_description_list)
        return centers

    def run(self, image):
        # Convert BGR (OpenCV) to RGB
        self.image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # Convert to base64 string
        buffered = BytesIO()
        self.image.save(buffered, format="JPEG")
        ollama_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # VLM
        print("Starting VLM")
        vlm = VLM(self.vlm_name, ollama_image)
        self.environment_description_list = vlm.run_and_parse()

        # Segmentation
        print("Starting Segmentation")
        self.centers_location = self.segmentation()
        self.environment_pos = {
            item: list(coord)
            for item, coord in zip(
                self.environment_description_list, self.centers_location
            )
        }
        return self.environment_description_list, vlm.raw_output, self.frame_with_masks_and_centers

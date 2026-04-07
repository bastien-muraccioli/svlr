from xml.parsers.expat import model
from src.vlm import VLM
from src.entity import Entity

from transformers import EdgeTamVideoModel, Sam2VideoProcessor
import torch

from tools.robot_tool import RobotCamera
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

TARGET_W = 640
TARGET_H = 360

def resize_frame_for_inference(frame):
    """
    Downscale frame to 640x360 for EdgeTAM/SAM2 inference,
    keep aspect ratio safe, and return:
    - small frame (for model)
    - original frame
    - scale ratios to convert masks back
    """
    orig_h, orig_w = frame.shape[:2]

    small = cv.resize(frame, (TARGET_W, TARGET_H), interpolation=cv.INTER_AREA)

    scale_x = orig_w / TARGET_W
    scale_y = orig_h / TARGET_H

    return small, frame, (scale_x, scale_y)

class Perception:
    def __init__(self, robot_camera: RobotCamera, vlm_name: str, vlm_provider: str):
        self.robot_camera = robot_camera
        self.vlm_name = vlm_name
        self.vlm_provider = vlm_provider
        if vlm_provider != "Ollama":
            raise NotImplementedError(
                f"VLM provider {vlm_provider} not implemented yet, only Ollama is supported."
            )
        self.seg_model_name = "language-segment-anything"

        self.environment_description_list = []  # entity class list
        self.mask_opacity = 0.2

        self.image = None
        self.frame_with_masks_and_centers = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.edgetam_model = EdgeTamVideoModel.from_pretrained("yonigozlan/EdgeTAM-hf").to(self.device, dtype=torch.float16)
        self.edgetam_processor = Sam2VideoProcessor.from_pretrained("yonigozlan/EdgeTAM-hf")
        self.inference_sessions = {}  # key: entity_name -> session

        self.session_reset_interval = 100  # reset every N frames to avoid memory leak
        self.local_frame_idx = {}  # key: entity_name -> local frame index
        self.edgetam_session = None
        
        self.lang_sam = LangSAM()


    def initialize_trackers(self):
        frame_np = np.array(self.image.convert("RGB")).copy()

        # Downscale for EdgeTAM
        small_frame, original_frame, (sx, sy) = resize_frame_for_inference(frame_np)

        for entity in self.environment_description_list:
            if not entity.found:
                continue

            session = self.edgetam_processor.init_video_session(
                inference_device=self.device,
                dtype=torch.float16
            )

            # Scale bbox to small frame
            x0, y0, x1, y1 = map(int, entity.bbox)
            x0 = int(x0 / sx)
            y0 = int(y0 / sy)
            x1 = int(x1 / sx)
            y1 = int(y1 / sy)
            input_boxes = [[[x0, y0, x1, y1]]]

            rgb_small_pil = Image.fromarray(small_frame)
            inputs = self.edgetam_processor(rgb_small_pil, device=self.device, return_tensors="pt")
            original_size = inputs.original_sizes[0]

            self.edgetam_processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=1,
                input_boxes=input_boxes,
                original_size=original_size,
            )

            self.inference_sessions[entity.name] = session
            self.local_frame_idx[entity.name] = 1
            entity.tracked = True

    def reset_all_session(self):
        """Safely resets all EdgeTAM sessions to prevent memory leaks."""
        for entity in self.environment_description_list:
            # Delete old session
            old_session = self.inference_sessions.get(entity.name, None)
            if old_session:
                del old_session
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    def _reset_session(self, entity, current_frame_np):
        """Safely resets an EdgeTAM session for a single entity."""
        print(f"[EdgeTAM] Resetting session for '{entity.name}' to prevent memory leak...")

        # Delete old session
        old_session = self.inference_sessions.get(entity.name, None)
        if old_session:
            del old_session

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Downscale frame
        small_frame, original_frame, (sx, sy) = resize_frame_for_inference(current_frame_np)

        # Recreate session
        session = self.edgetam_processor.init_video_session(
            inference_device=self.device,
            dtype=torch.float16,
        )

        # Scale bbox to small frame
        x0, y0, x1, y1 = map(int, entity.bbox)
        x0 = int(x0 / sx)
        y0 = int(y0 / sy)
        x1 = int(x1 / sx)
        y1 = int(y1 / sy)
        input_boxes = [[[x0, y0, x1, y1]]]

        rgb_small_pil = Image.fromarray(small_frame)
        inputs = self.edgetam_processor(rgb_small_pil, device=self.device, return_tensors="pt")
        original_size = inputs.original_sizes[0]

        self.edgetam_processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=1,
            input_boxes=input_boxes,
            original_size=original_size,
        )

        self.inference_sessions[entity.name] = session
        self.local_frame_idx[entity.name] = 1



    def update_trackers(self, frame):
        frame_np = np.array(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))).copy()

        # Downscale for EdgeTAM
        small_frame, original_frame, (sx, sy) = resize_frame_for_inference(frame_np)

        for entity in self.environment_description_list:
            if not entity.found or not entity.tracked:
                continue

            session = self.inference_sessions.get(entity.name, None)
            if session is None:
                continue

            # Handle VRAM-safe session reset
            if self.local_frame_idx[entity.name] % self.session_reset_interval == 0:
                self._reset_session(entity, frame_np)
                session = self.inference_sessions[entity.name]

            # Prepare input
            rgb_small_pil = Image.fromarray(small_frame)
            with torch.inference_mode():
                inputs = self.edgetam_processor(
                    images=rgb_small_pil, device=self.device, return_tensors="pt"
                )

                pixel_values = inputs.pixel_values[0].half().contiguous()

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.edgetam_model(
                        inference_session=session,
                        frame=pixel_values
                    )

                mask_tensor = self.edgetam_processor.post_process_masks(
                    [outputs.pred_masks],
                    original_sizes=inputs.original_sizes,
                    binarize=True
                )[0]

            # Mask to numpy
            mask_np = mask_tensor.squeeze().cpu().numpy()
            if mask_np.size == 0 or np.isnan(mask_np).any():
                mask_np = np.zeros(small_frame.shape[:2], dtype=np.uint8)
            else:
                mask_np = (mask_np * 255).astype(np.uint8)

            # Upscale mask back to original frame
            mask_np = cv.resize(mask_np, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv.INTER_NEAREST)

            # Compute centroid and bbox
            centroid, bbox = self.centroid_segmentation(mask_np)
            if centroid is not None:
                entity.update_position((int(centroid[0]), int(centroid[1])))
                entity.bbox = bbox
                entity.mask = mask_np
            else:
                entity.reset_tracking()
                self.inference_sessions.pop(entity.name, None)

            # increment local index
            self.local_frame_idx[entity.name] += 1

        # Build visualization
        self.build_frame_with_masks_and_centers(original_frame)
        return self.environment_description_list, cv.cvtColor(
            self.frame_with_masks_and_centers, cv.COLOR_RGB2BGR
        )



    def build_frame_with_masks_and_centers(self, original_frame):
        """
        Combine original frame + yellow semi-transparent masks + red centroids with labels.
        """

        masks = []
        labels = []
        bboxes = []
        centers = []
        robot_coords = []
        for entity in self.environment_description_list:
            if not entity.found or entity.mask is None or not entity.tracked:
                continue
            masks.append(entity.mask)
            labels.append(entity.name)
            centers.append(entity.pixel_pos)
            bboxes.append(entity.bbox)
            robot_coords.append(entity.robot_frame_pos)
            # print(f"Entity '{entity.name}' at pixel {entity.pixel_pos} and robot frame {entity.robot_frame_pos}")

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
        
        # Draw bounding boxes, centroids and labels
        for bbox, label, (x, y), robot_coord in zip(bboxes, labels, centers, robot_coords):
            x0, y0, x1, y1 = map(int, bbox)
            cv.rectangle(blended, (x0, y0), (x1, y1), (0, 255, 255), 2)  # yellow box
            x, y = int(x), int(y)
            cv.circle(blended, (x, y), 6, (0, 0, 255), -1)  # red dot
            cv.putText(
                blended,
                label,
                (x + 10, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            coord_text = f"({robot_coord[0]*1000:.0f}, {robot_coord[1]*1000:.0f}, {robot_coord[2]*1000:.0f})mm"
            wrapped_text = textwrap.fill(coord_text, width=30)
            cv.putText(
                blended,
                wrapped_text,
                (x + 10, y + 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
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

        # Convert PIL to RGB and keep original NumPy for processing
        image_pil = self.image.convert("RGB").copy()
        image_np = np.array(image_pil)

        imgs_seg = []
        centers  = []
        bboxes   = []  # store bounding boxes from segmentation

        # Loop over each entity individually
        for entity in self.environment_description_list:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = self.lang_sam.predict([image_pil], [entity.name])[0]

            # Take only the highest‑confidence mask
            masks  = result["masks"]
            scores = result["scores"]
            if scores is None or np.size(scores) == 0 or masks is None or len(masks) == 0:
                print(f"No segmentation found for '{entity.name}', skipping.")
                self.environment_description_list.remove(entity)
                continue

            best_idx = int(np.argmax(scores))
            best_mask = (masks[best_idx].astype(np.uint8) * 255)

            # Compute centroid & bbox
            center, bbox = self.centroid_segmentation(best_mask)
            if center is None or bbox is None:
                self.environment_description_list.remove(entity)
                continue

            # Store 2D center
            centers.append([center[0], center[1]])
            bboxes.append(bbox)  # save bbox for tracker initialization

            # Draw box+center on mask for visualization
            vis = best_mask.copy()
            x0, y0, x1, y1 = map(int, bbox)
            cv.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 3)
            cv.circle(vis, (int(center[0]), int(center[1])), 5, (255,255,255), -1)
            imgs_seg.append(vis)

        if not centers:
            print("No objects found in segmentation.")
            return

        # Rescale centers and bboxes back to original image dimensions
        orig_w, orig_h = self.image.size
        mask_h, mask_w = (imgs_seg[0].shape[:2] if imgs_seg else (1,1))
        final_img = image_np.copy()
        for i, c in enumerate(centers):
            c[0] = c[0] * orig_w / mask_w
            c[1] = c[1] * orig_h / mask_h
            # Scale bounding boxes
            x0, y0, x1, y1 = bboxes[i]
            bboxes[i] = [
                x0 * orig_w / mask_w,
                y0 * orig_h / mask_h,
                x1 * orig_w / mask_w,
                y1 * orig_h / mask_h,
            ]
            cv.circle(final_img, (int(c[0]), int(c[1])), 20, (255, 0, 0), -1)
        
        # Update entity positions, masks and bboxes
        for i, entity in enumerate(self.environment_description_list):
            entity.update_position((int(centers[i][0]), int(centers[i][1])))
            entity.bbox = bboxes[i]
            entity.mask = imgs_seg[i]           

        # Build composite frame with masks and centers
        self.build_frame_with_masks_and_centers(image_np)

    def segment_one_entity(self, entity_name: str):
        """
         Segment one entity from the image and add it to the environment description list.
         Returns True if successful, False otherwise.
        """
        print(f"Run Image Segmentation model {self.seg_model_name} for entity '{entity_name}'")

        # Convert PIL to RGB and keep original NumPy for processing
        image_pil = self.image.convert("RGB").copy()
        # image_np = np.array(image_pil)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            result = self.lang_sam.predict([image_pil], [entity_name])[0]

        # Take only the highest‑confidence mask
        masks  = result["masks"]
        scores = result["scores"]
        if scores is None or np.size(scores) == 0 or masks is None or len(masks) == 0:
            print(f"No segmentation found for '{entity_name}', skipping.")
            return False

        best_idx = int(np.argmax(scores))
        best_mask = (masks[best_idx].astype(np.uint8) * 255)

        # Compute centroid & bbox
        center, bbox = self.centroid_segmentation(best_mask)
        if center is None or bbox is None:
            return False
        # Store 2D center
        center[0] = center[0] * self.image.size[0] / best_mask.shape[1]
        center[1] = center[1] * self.image.size[1] / best_mask.shape[0]
        # Scale bounding boxes
        x0, y0, x1, y1 = bbox
        bbox = [
            x0 * self.image.size[0] / best_mask.shape[1],
            y0 * self.image.size[1] / best_mask.shape[0],
            x1 * self.image.size[0] / best_mask.shape[1],
            y1 * self.image.size[1] / best_mask.shape[0],
        ]

        # After computing mask and bbox
        entity = Entity(entity_name, self.robot_camera)
        entity.update_position((int(center[0]), int(center[1])))
        entity.mask = best_mask

        # Initialize EdgeTam session
        session = self.edgetam_processor.init_video_session(
            inference_device=self.device,
            dtype=torch.float16
        )
        inputs = self.edgetam_processor(self.image.convert("RGB"), device=self.device, return_tensors="pt")
        original_size = inputs.original_sizes[0]
        x0, y0, x1, y1 = map(int, bbox)
        input_boxes = [[[x0, y0, x1, y1]]]
        self.edgetam_processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=1,
            input_boxes=input_boxes,
            original_size=original_size,
        )
        self.inference_sessions[entity_name] = session
        entity.tracked = True
        entity.found = True

        #  Check if entity with same name already exists, if so replace it
        for i, existing_entity in enumerate(self.environment_description_list):
            if existing_entity.name == entity_name:
                self.environment_description_list[i] = entity
                return True

        self.environment_description_list.append(entity)

        return True


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
        entities_name_found = vlm.run_and_parse()
        self.environment_description_list = [Entity(name, self.robot_camera) for name in entities_name_found]

        # Segmentation
        print("Starting Segmentation")
        self.segmentation()

        # Initialize trackers
        self.initialize_trackers()

        return self.environment_description_list, vlm.raw_output, cv.cvtColor(self.frame_with_masks_and_centers, cv.COLOR_RGB2BGR)

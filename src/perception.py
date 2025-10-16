from src.vlm import VLM
from src.entity import Entity

# import torch
# from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
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
        self.mask_opacity = 0.1

        self.image = None
        self.frame_with_masks_and_centers = None

    def initialize_trackers(self):
        """
        Initialize CSRT trackers for all detected objects after segmentation.
        """
        frame_np = np.array(self.image.convert("RGB"))
        # We need the original masks to compute accurate bounding boxes
        for entity in self.environment_description_list:
            if not entity.found:
                continue
            x0, y0, x1, y1 = map(int, entity.bbox)
            w = x1 - x0
            h = y1 - y0
            tracker_bbox = (x0, y0, w, h)

            tracker = cv.TrackerCSRT_create()
            tracker.init(frame_np, tracker_bbox)
            entity.tracker = tracker
            entity.bbox = tracker_bbox
            entity.tracked = True

    def update_trackers(self, frame):
        """
        Update CSRT trackers for the current frame.
        Updates centers_location and frame_with_masks_and_centers.
        """
        frame_PIL = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        frame_np = np.array(frame_PIL.convert("RGB"))

        for entity in self.environment_description_list:
            if not entity.found or not entity.tracked:
                continue

            success, bbox = entity.tracker.update(frame_np)
            if not success:
                print(f"Tracking failed for {entity.name}, removing.")
                entity.found = False
                entity.reset_tracking()
                continue

            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            xf = x+w
            yf = y+h
            entity.update_position((int(cx), int(cy)))
            entity.bbox = [x, y, xf, yf]
            mask = np.zeros(frame_np.shape[:2], dtype=np.uint8)
            mask[y:yf, x:xf] = 255
            entity.mask = mask


        # Build frame visualization with updated bounding boxes
        self.build_frame_with_masks_and_centers(frame_np)
        return self.environment_description_list, cv.cvtColor(self.frame_with_masks_and_centers, cv.COLOR_RGB2BGR)


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
        # Initialize model
        model = LangSAM()

        # Convert PIL to RGB and keep original NumPy for processing
        image_pil = self.image.convert("RGB")
        image_np = np.array(image_pil)

        imgs_seg = []
        centers  = []
        bboxes   = []  # store bounding boxes from segmentation

        # Loop over each entity individually
        for entity in self.environment_description_list:
            result = model.predict([image_pil], [entity.name])[0]

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
        # Initialize model
        model = LangSAM()

        # Convert PIL to RGB and keep original NumPy for processing
        image_pil = self.image.convert("RGB")
        image_np = np.array(image_pil)

        result = model.predict([image_pil], [entity_name])[0]

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

        entity = Entity(entity_name, self.robot_camera)
        entity.update_position((int(center[0]), int(center[1])))
        entity.mask = best_mask

        x0, y0, x1, y1 = map(int, bbox)
        w = x1 - x0
        h = y1 - y0
        tracker_bbox = (x0, y0, w, h)

        tracker = cv.TrackerCSRT_create()
        tracker.init(image_np, tracker_bbox)
        entity.tracker = tracker
        entity.bbox = tracker_bbox
        entity.tracked = True
        entity.found = True
        entity.need_to_be_tracked = False  # Already found

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

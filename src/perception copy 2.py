from src.vlm import VLM

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
    def __init__(self, vlm_name: str, vlm_provider: str):
        self.vlm_name = vlm_name
        self.vlm_provider = vlm_provider
        if vlm_provider != "Ollama":
            raise NotImplementedError(
                f"VLM provider {vlm_provider} not implemented yet, only Ollama is supported."
            )
        self.seg_model_name = "language-segment-anything"

        self.environment_description_list = []  # ["figurine", "cup", "table"]
        self.centers_location = []  # [(x1,y1,z1), ...]
        self.environment_pos = {}  # {'figurine':[x,y,z], ...}
        self.mask_opacity = 0.2

        self.image = None
        self.frame_with_masks_and_centers = None

        # --- TRACKING state ---
        # tracked_objects: list of dict { label, mask (uint8 same size as frames), center, bbox, lost_count }
        self.tracked_objects = []
        self.prev_gray = None
        self.max_lost_frames = 5     # remove object after this many consecutive frames with no detection
        self.min_mask_area = 50      # if warped mask area < this -> considered lost (tune to your scale)

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
        """
        Run LangSAM segmentation per prompt and return centers list (2D + dummy Z=0).
        Also builds imgs_seg (list of binary masks same size as original image).
        """
        print(f"Run Image Segmentation model {self.seg_model_name}")
        model = LangSAM()

        # Convert PIL to RGB and keep original NumPy for processing
        image_pil = self.image.convert("RGB")
        image_np = np.array(image_pil)

        imgs_seg = []
        centers = []
        not_found = []

        for i, prompt in enumerate(self.environment_description_list):
            result = model.predict([image_pil], [prompt])[0]
            masks = result["masks"]
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

            centers.append([center[0], center[1], 0])

            # Draw box+center on mask for visualization
            vis = best_mask.copy()
            x0, y0, x1, y1 = map(int, bbox)
            cv.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 3)
            cv.circle(vis, (int(center[0]), int(center[1])), 5, (255, 255, 255), -1)
            imgs_seg.append(vis)

        # Remove not-found prompts
        for nf in not_found:
            print(f"Object: {nf} not found, removing from descriptions")
            self.environment_description_list.remove(nf)

        # Rescale centers to original image size
        orig_w, orig_h = self.image.size
        mask_h, mask_w = (imgs_seg[0].shape[:2] if imgs_seg else (1, 1))
        for c in centers:
            c[0] = c[0] * orig_w / mask_w
            c[1] = c[1] * orig_h / mask_h

        # Draw final centers on a copy of the original
        final_img = image_np.copy()
        for x, y, _ in centers:
            cv.circle(final_img, (int(x), int(y)), 20, (255, 0, 0), -1)

        # Build composite frame
        self.build_frame_with_masks_and_centers(image_np, imgs_seg, centers, self.environment_description_list)

        # Return centers and masks for tracking initialization
        return centers, imgs_seg

    # ------------------ TRACKING METHODS ------------------

    def start_tracking_from_masks(self, initial_frame_bgr: np.ndarray, masks: list, centers: list, labels: list):
        """
        Initialize the tracker using initial frame, masks, centers, and labels.
        - initial_frame_bgr: frame in BGR uint8 (as your run() provides)
        - masks: list of binary masks (uint8 0..255) same size as initial frame
        - centers: list of [x,y,0] in image coordinates (float)
        - labels: list of strings (prompts)
        This populates self.tracked_objects and stores prev_gray for optical flow.
        """
        frame = initial_frame_bgr.copy()
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame.shape[2] == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray = frame  # unlikely but safe

        self.prev_gray = gray
        self.tracked_objects = []

        img_h, img_w = gray.shape[:2]

        # Ensure masks are the same size as frame and binary uint8
        prepared_masks = []
        for m in masks:
            if m is None:
                prepared_masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
                continue
            mm = m.copy()
            if mm.shape[:2] != (img_h, img_w):
                mm = cv.resize(mm, (img_w, img_h), interpolation=cv.INTER_NEAREST)
            if mm.max() <= 1:
                mm = (mm * 255).astype(np.uint8)
            prepared_masks.append((mm > 127).astype(np.uint8) * 255)

        # Build tracked_objects entries
        for label, mask, center in zip(labels, prepared_masks, centers):
            # compute bbox using contours
            contours, _ = cv.findContours((mask > 127).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(cnt)
                bbox = [x, y, x + w, y + h]
            else:
                bbox = [0, 0, img_w - 1, img_h - 1]

            obj = {
                "label": label,
                "mask": mask,         # uint8 0..255
                "center": [center[0], center[1]],  # x,y
                "bbox": bbox,
                "lost_count": 0
            }
            self.tracked_objects.append(obj)

        # update other state
        self._update_public_states_from_tracked()

    def update_tracking_frame(self, new_frame_bgr: np.ndarray):
        """
        Propagate tracked masks using optical flow (Farneback) to the new frame.
        Updates centroids, bounding boxes, and removes lost objects.
        Returns: (environment_description_list, None, frame_with_masks_and_centers)
        """
        if self.prev_gray is None or not self.tracked_objects:
            raise RuntimeError("Tracker not initialized. Call start_tracking_from_masks() first.")

        frame = new_frame_bgr.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # --- Equalize for better optical flow stability ---
        prev_eq = cv.equalizeHist(self.prev_gray)
        gray_eq = cv.equalizeHist(gray)

        # --- Dense optical flow from previous frame to current frame ---
        flow = cv.calcOpticalFlowFarneback(prev_eq, gray_eq, None,
                                        pyr_scale=0.5, levels=5,
                                        winsize=25, iterations=5,
                                        poly_n=7, poly_sigma=1.5, flags=0)

        h, w = gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # --- Correct direction: warp previous mask forward to current frame ---
        map_x = (grid_x - flow[..., 0]).astype(np.float32)
        map_y = (grid_y - flow[..., 1]).astype(np.float32)

        updated_objects = []
        removed_labels = []

        for obj in self.tracked_objects:
            old_mask = (obj["mask"] > 127).astype(np.uint8) * 255
            warped = cv.remap(old_mask, map_x, map_y,
                            interpolation=cv.INTER_NEAREST,
                            borderMode=cv.BORDER_CONSTANT, borderValue=0)

            # Small morphological smoothing to reduce fragmentation
            if warped.max() > 0:
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
                warped = cv.morphologyEx(warped, cv.MORPH_OPEN, kernel)

            area = int((warped > 127).sum())
            if area < 20:  # object might be lost
                obj["lost_count"] += 1
                if obj["lost_count"] > 3:
                    removed_labels.append(obj["label"])
                    continue
            else:
                obj["lost_count"] = 0
                centroid, bbox = self.centroid_segmentation(warped)
                if centroid is not None:
                    obj["center"] = [centroid[0], centroid[1]]
                    obj["bbox"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    obj["mask"] = warped

            updated_objects.append(obj)

        # Replace tracked objects
        self.tracked_objects = updated_objects

        # Remove lost objects from environment_description_list and environment_pos
        for lab in removed_labels:
            if lab in self.environment_description_list:
                self.environment_description_list.remove(lab)
            if lab in self.environment_pos:
                del self.environment_pos[lab]

        # Update previous gray frame
        self.prev_gray = gray

        # Update public states (centers_location, environment_pos)
        self._update_public_states_from_tracked()

        # Build visualization frame
        masks_vis = [obj["mask"] for obj in self.tracked_objects]
        centers_vis = [[obj["center"][0], obj["center"][1], 0] for obj in self.tracked_objects]
        labels_vis = [obj["label"] for obj in self.tracked_objects]
        self.build_frame_with_masks_and_centers(frame, masks_vis, centers_vis, labels_vis)

        return self.environment_description_list, None, self.frame_with_masks_and_centers



    def _update_public_states_from_tracked(self):
        """
        Keep centers_location and environment_pos consistent with tracked_objects.
        """
        centers_location = []
        env_pos = {}
        for obj in self.tracked_objects:
            cx, cy = obj["center"]
            centers_location.append([cx, cy, 0])
            env_pos[obj["label"]] = [cx, cy, 0]

        self.centers_location = centers_location
        self.environment_pos = env_pos

    # ------------------ ORIGINAL run() with small change ------------------

    def run(self, image):
        """
        Initial run for first frame:
        - sets self.image
        - queries VLM for descriptions
        - runs segmentation to produce initial masks + centers
        - initializes the optical-flow tracker using those masks
        """
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
        centers, imgs_seg = self.segmentation()
        # segmentation() already removed not-found prompts

        # Initialize tracking with masks and centers
        # image is passed as BGR to be consistent with how frames come in
        self.start_tracking_from_masks(image, imgs_seg, centers, self.environment_description_list)

        return self.environment_description_list, vlm.raw_output, self.frame_with_masks_and_centers

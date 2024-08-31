from src.vlm import VLM

import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
import skimage.measure as sim
import skimage.transform as sit
import re 
import os
import threading
import textwrap

def save_image(image_path: str, image_data):
    cv.imwrite(image_path, image_data)
    

def parse_vlm_output(text):
        text = text.strip().replace("-", "")
        text_list = text.split("\n")
        text_list = [text.strip() for text in text_list]

        return text_list
    
class Perception:
    def __init__(self):
        self.vlm_name = "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"
        self.seg_model_name = "CIDAS/clipseg-rd64-refined"
        self.pictures_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pictures")
        self.seg_result_image_path = os.path.join(self.pictures_folder_path, "seg_result.png") # Result of the centroid segmentation
        self.plot_image_path = os.path.join(self.pictures_folder_path, "plot.png")
        
        self.environment_description_list = [] # ["figurine", "cup", "table"]
        self.centers_location = [] # [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
        self.environment_pos = {} # {'figurine':[x1,y1,z1], 'cup':[x2,y2,z2], 'table':[x3,y3,z3]}
        self.image = None

    

    def centroid_segmentation(self, map):
        """
        Connected component analysis to find the region with the highest median value, and return the centroid of each region.
        """
        resize_image_format = (64, 64)
        im_resized = sit.resize(map, resize_image_format)
         # Apply gaussian blur, threshold and finally labeling the image
        label_map = sim.label(cv.GaussianBlur(im_resized, ksize=(5, 5), sigmaX=1) > 0.3)

        #segmentation
        regions = sim.regionprops(label_map)
        
        # If no regions are found, return None
        if regions == []:
            return None, None

        #find the region with the highest median value
        
        median, best_region = -1, None
        for region in regions:
            values = im_resized[region.coords[:,0], region.coords[:,1]]
            if np.median(values) > median:
                median = np.median(values)
                best_region = region
        
        # Calculate the centroid and bounding box of the best region 
        centroid = list(best_region.centroid)
        centroid[0] *= map.shape[0]/resize_image_format[0]; centroid[1] *= map.shape[1]/resize_image_format[1]

        bbox = list(best_region.bbox)
        bbox[0] *= map.shape[0] / resize_image_format[0]; bbox[2] *= map.shape[0] / resize_image_format[0]
        bbox[1] *= map.shape[1] / resize_image_format[1]; bbox[3] *= map.shape[1] / resize_image_format[1]

        centroid = centroid[::-1]
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

        return centroid, bbox
        # return centroid
    
    def segmentation(self):      
        # Load model
        print(f"Run Image Segmentation model {self.seg_model_name}")
        processor = CLIPSegProcessor.from_pretrained(self.seg_model_name)
        model = CLIPSegForImageSegmentation.from_pretrained(self.seg_model_name)

        # Prepare image & texts for model
        inputs = processor(text=self.environment_description_list, images=[self.image] * len(self.environment_description_list), padding="max_length", return_tensors="pt")

        # Forward pass and visualize the predictions of the model
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs.logits.unsqueeze(1)
        
        imgs_seg = []
        centers = []
        environment_not_found = [] # List of indexes of objects that were not found in the segmentation
        
        # # Compute center and bounding box + Visualize each object 
        for i in range(len(self.environment_description_list)):
                        
            # Convert tensor to int8 image
            img_array = torch.sigmoid(preds[i][0]).cpu().numpy()
            img_array = (img_array * 255).astype(np.uint8)

            # Save the image in a thread using OpenCV
            img_path = os.path.join(self.pictures_folder_path , f"prediction_{i}.jpg")
            thread = threading.Thread(target=save_image, args=(img_path, img_array))
            thread.start()

            # Compute bounding box and centroids
            center, bbox = self.centroid_segmentation(img_array)
            
            # If no center or bbox is found, remove the object
            if center is None or bbox is None:
                environment_not_found.append(self.environment_description_list[i])
                continue
            
            min_x, min_y, max_x, max_y = bbox
            min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)

            center_x, center_y = center
            #For the moment we fix the Z manually (no depth with camera)
            center_z = 0
            centers.append([center_x, center_y, center_z])

            # Set up to visualize bounding box and its center
            cv.rectangle(img_array,(min_x,min_y),(max_x,max_y),(255,255,255),3)
            cv.circle(img_array, (int(center_x), int(center_y)), radius=5, color=(255, 255, 255), thickness=-1)
            imgs_seg.append(img_array)                

        # Remove objects that were not found in the segmentation
        for item in environment_not_found:
            print(f"Object: {item} not found, removing it from the environment description list")
            self.environment_description_list.remove(item)
        
        image_shape = self.image.size[:2]
        resized_shape = img_array.shape[:2]
        for i in range(len(centers)):
            x, y, _ = centers[i]
            centers[i][0] = x * image_shape[0] / resized_shape[0]
            centers[i][1] = y * image_shape[1] / resized_shape[1]
        img_array = np.array(self.image)
        for c in centers:
            cv.circle(img_array, (int(c[0]), int(c[1])), radius=20, color=(255,0,0), thickness=-1)

        # Show plot
        thread_plot = threading.Thread(target=self.generate_plot, args=(imgs_seg, img_array))
        thread_plot.start()

        return centers
    
    def generate_plot(self, imgs_seg, result_img):
        num_plots = len(imgs_seg) + 2
        
        # Create a single figure with subplots
        fig, ax = plt.subplots(1, num_plots, figsize=(14, 5))
        
        # Adjust the vertical position of the title
        fig.suptitle("Image captured and segmentation results based on the VLM outputs", fontsize=16, y=0.92)
        
        # Turn off axis for all subplots
        for a in ax:
            a.axis('off')

        # Display the base image in the first subplot
        ax[0].imshow(self.image)
        ax[0].set_title('Base Image', fontsize=14)
        
        # Display the result image in the last subplot
        ax[num_plots-1].imshow(result_img)
        ax[num_plots-1].set_title('Result image with centroids', fontsize=14)
        
        # Display each segmented image and add text below
        for i, img in enumerate(imgs_seg):
            ax[i + 1].imshow(img)
            ax[i + 1].set_title(f'Segment {i + 1}', fontsize=14)
            ax[i + 1].axis('off')
            
            
            # Create a text annotation below the image
            text = self.environment_description_list[i]
            wrapped_text = textwrap.fill(text, width=30)
            ax[i + 1].annotate(
                wrapped_text,
                xy=(0.5, -0.1),  # Position of the text below the image
                xycoords='axes fraction',
                fontsize=10,
                ha='center',
                va='top',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'),
                color='white'
            )

        # Manually adjust subplot parameters to reduce blank space
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.2)  # Adjust margins to minimize blank space
        
        # Save the plot with tight bounding box to avoid excess white space
        plt.savefig(self.plot_image_path, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close()
            
        

    def run(self, image):
        self.image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        # VLM
        print("Starting VLM")
        vlm = VLM(self.vlm_name, self.image)
        vlm_output = vlm.run()
        self.environment_description_list = parse_vlm_output(vlm_output)

        #Segmentation
        print("Starting Segmentation")
        self.centers_location = self.segmentation()
        self.environment_pos = {item: list(coord) for item, coord in zip(self.environment_description_list, self.centers_location)}

        return self.environment_description_list
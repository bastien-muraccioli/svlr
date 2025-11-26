import requests
import json
import time

class VLM:
    def __init__(self, vlm_name: str, image=None):

        self.name = vlm_name
        self.raw_output = ""
        self.image = image

        self.prompt = \
"""You see a top-down view of an image.
Identify every distinct physical object or entity visible in the image.
List each object individually. If multiple similar or identical objects appear (e.g., two cups, two photos, or two apples), list each one as a separate entry.
Do not group or merge objects under a single description. For example, output ["photo of a man", "photo of a cat"] instead of ["photos of a man and a cat"].
Each item should represent a single, identifiable thing (e.g., “red ceramic mug”, “white table”, “metal spoon”).
Be concise but descriptive.
Do not include any explanatory text or formatting other than JSON.
Respond only as a valid JSON array of strings, one string per entity. 
Example:
["white ceramic cup", "silver spoon", "brown wooden table"]
"""

    def run(self):
        # single-round single-image conversation
        start = time.time()
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": self.name,
            "prompt": self.prompt,
            "images": [self.image],
        }, stream=True)
        output = ""
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        output += data.get("response", "")
                    except json.JSONDecodeError:
                        print("Could not decode:", line)
            # print("Answer:", output.strip())
        else:
            print("Request failed with status", response.status_code)
            print(response.text)
        end = time.time()
        print(f"VLM {self.name} Inference time = {end - start}s")
        print(f"VLM Prompt = {self.prompt}")
        print(f"VLM Response = {output.strip()}")
        #  Unload the model to save memory
        requests.post("http://localhost:11434/api/generate", json={
            "model": self.name,
            "keep_alive": 0})
        return output.strip()
    
    def use_as_a_llm(self, prompt: str):
        start = time.time()
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": self.name,
            "prompt": prompt,
        }, stream=True)
        output = ""
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        output += data.get("response", "")
                    except json.JSONDecodeError:
                        print("Could not decode:", line)
            # print("Answer:", output.strip())
        else:
            print("Request failed with status", response.status_code)
            print(response.text)
        end = time.time()
        print(f"VLM {self.name} Inference time = {end - start}s")
        print(f"VLM Prompt = {self.prompt}")
        print(f"VLM Response = {output.strip()}")
        #  Unload the model to save memory
        requests.post("http://localhost:11434/api/generate", json={
            "model": self.name,
            "keep_alive": 0})
        return output.strip()

    def parse_vlm_output(self):
        """
        Parse the JSON-style output from a VLM into a clean list of strings.

        Args:
            text (str): The VLM response, expected to be a JSON array of strings.

        Returns:
            list[str]: A list of trimmed object descriptions.
        """
        try:
            # Attempt to parse as JSON
            text_list = json.loads(self.raw_output)
            
            # Ensure it's a list of strings
            if isinstance(text_list, list):
                text_list = [str(t).strip() for t in text_list if isinstance(t, (str, int, float))]
                return text_list
            else:
                raise ValueError("Parsed JSON is not a list.")
        
        except json.JSONDecodeError:
            # Fallback: Try to recover if brackets or quotes are missing
            import re
            items = re.findall(r'"(.*?)"|\'(.*?)\'', self.raw_output)
            text_list = [a or b for a, b in items]
            return [t.strip() for t in text_list if t.strip()]
        
    def run_and_parse(self):
        self.raw_output = self.run()
        parsed_output = self.parse_vlm_output()
        print(f"VLM Parsed Output = {parsed_output}")
        return parsed_output

# CXR LLaVA
### Multimodal Large Language Model Fine-Tuned for Chest X-ray Images
CXR LLaVA is an innovative open-source, multimodal large language model specifically designed for generating radiologic reports from chest X-ray images.

-   **Arxiv Preprint Paper**: Explore the detailed scientific background of CXR LLaVA on [Arxiv](https://arxiv.org/abs/2310.18341).
-   **Demo Website**: Experience the model in action at [Radiologist App](https://radiologist.app/cxr-llava).


|Version| Input CXR resolution | Channels | Vision Encoder | Base LLM | Weight 
|--|--|--|--|--|--|
| v1.0 | 512x512 | RGB|RN50|LLAMA2-13B-CHAT|Deprecated
|v2.0 (Latest)|512x512|Grayscale|ViT-L/16|LLAMA2-7B-CHAT| <a href="https://huggingface.co/ECOFRI/CXR-LLAVA-v2" target="_blank">Link</a>

You can interpret CXR with just 6 lines of code. 

(NVIDIA GPU VRAM>=14GB needed)
```python
from transformers import AutoModel
from PIL import Image
model = AutoModel.from_pretrained("ECOFRI/CXR-LLAVA-v2", trust_remote_code=True)
model = model.to("cuda")
cxr_image = Image.open(os.path.join(os.path.dirname(__file__), "IMG", "img.jpg"))
response = model.write_radiologic_report(cxr_image)
```
 > The radiologic report reveals a large consolidation in the right upper lobe of the lungs. There is no evidence of pleural effusion or pneumothorax. The cardiac and mediastinal contours are normal. 


## Usage Guide
### Importing Packages
```python
from transformers import AutoModel
from PIL import Image
```
### Prepare CXR
    
<img src="/IMG/img.jpg"  width="300"></img><br/>

Ensure you have an CXR image file ready, such as 'img.jpg'.

Use the following code to load the image
```python
cxr_image = Image.open(os.path.join(os.path.dirname(__file__), "IMG", "img.jpg"))
```
### Load model
Loading the CXR-LLAVA model is straightforward and can be done in one line of code.

```python
model = AutoModel.from_pretrained("ECOFRI/CXR-LLAVA-v2", trust_remote_code=True)
model = model.to("cuda")
```

### Generating Radiologic Reports

To write a radiologic report of a chest radiograph:


```python
response = model.write_radiologic_report(cxr_image)
```

 > The radiologic report reveals a large consolidation in the right upper lobe of the lungs. There is no evidence of pleural effusion or pneumothorax. The cardiac and mediastinal contours are normal. 


### Differential Diagnosis
For differential diagnosis:

```python
model.write_differential_diagnosis(cxr_image)
```
> Possible differential diagnoses for this patient include pneumonia,tuberculosis, lung abscess, or a neoplastic process such as lung cancer.

### Question Answering
To ask a question:
```python
question = "What is true meaning of consolidation?"
response = model.ask_question(question=question, image=cxr_image)
```
> Consolidation refers to the filling of the airspaces in the lungs with fluid, pus, blood, cells or other substances, resulting in a region of lung tissue that has become dense and solid rather than containing air.

### Custom Prompt
For custom interactions:
```python
img = Image.open(os.path.join(os.path.dirname(__file__), "IMG", "img.jpg"))
chat = [
    {"role": "system",
     "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
    {"role": "user",
     "content": "<image>\nWrite a radiologic report on the given chest radiograph, including information about atelectasis, cardiomegaly, consolidation, pulmonary edema, pleural effusion, and pneumothorax.\n"}
]
response = model.generate_cxr_repsonse(chat=chat,pil_image=img, temperature=0, top_p=1)
```

## License Information

CXR LLaVA is available under a Creative Commons NonCommercial License. 

Users must obtain the LLAMA-2 license prior to use. More details can be found [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).


Lastly, we extend our heartfelt thanks to all the contributors of the [LLaVA project](https://llava-vl.github.io/). 

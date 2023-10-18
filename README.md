# CXR LLaVA
### Multimodal Large Language Model Fine-Tuned for Chest X-ray Images

This study aimed to develop open-source multimodal large language model for chest X-ray images, potentially assisting the image interpretation of human radiologists. 

Original article is under review.

# Information
* **Base LLM** : LLAMA2-13B
* **Image Encoder** : Resnet50
* **Dataset** : Ensemble of multiple datasets including MIMIC. For more information, check our publication [under review]
* **Model Weight** : [uploading](https://google.com)

# Examples
<img src="/IMG/img.jpg" width="450px" ></img><br/>

**Question**

    Write a radiologic report on the given chest radiograph, including information about atelectasis, cardiomegaly, consolidation, pulmonary edema, pleural effusion, and pneumothorax.

**LLM output**

    The chest radiograph shows a large, rounded opacity in the right upper lobe. There is also a small right pleural effusion. The left lung is clear and there is no pneumothorax. The cardiomediastinal silhouette is normal. 

**Question**

    What is possible diagnosis?

**LLM output**

    Possible diagnosis includes pneumonia, lung cancer, or tuberculosis. 

**Question**

    Should additional radiologic study needed?

**LLM output**

    Further radiologic studies are recommended to confirm the diagnosis. A CT scan of the chest may provide more detailed information about the nature and extent of the opacity. 


# Requirement
## Hardware
You need a NVIDIA GPU with VRAM larger than 26GB. 
Recommended : NVIDIA-A100 40GB

## Software Dependency
torch, transformers==4.34.0, open_clip_torch==2.22.0
Refer to requirements.txt

## Code
```python
loader = CXR_LLAMA_Loader(model_path=model_path, temperature=0, top_p=0.7)
img = Image.open(os.path.join(os.path.dirname(__file__),"IMG","pneumonia.jpg"))
chat = [
    {"role": "system", "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
    {"role": "user", "content": "<image> Carefully read the provided chest x-ray and write a detailed radiological report."}
]

response = loader.generate(chat,pil_image=img)
print("QUESTION : %s"%chat[-1]['content'])
print("RESPONSE : %s"%response)

chat.append({"role":"assistant","content":response})
chat.append({"role":"user","content":"What is possible diagnosis?"})

response = loader.generate(chat,pil_image=img)

print("QUESTION : %s"%chat[-1]['content'])
print("RESPONSE : %s"%response)

chat.append({"role": "assistant", "content": response})
chat.append({"role": "user", "content": "Should additional radiologic study needed?"})

response = loader.generate(chat,pil_image=img)

print("QUESTION : %s"%chat[-1]['content'])
print("RESPONSE : %s"%response)
```





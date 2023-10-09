# CXR_LLAMA
A multimodal LLM interpreting chest radiograph trained by Radiologists.

We developed multimodal large language model based on Meta's LLAMA2 model that enable an enhanced approach to medical diagnostics by interpreting chest X-rays and generating detailed reports while also providing interactive querying capabilities, thereby assisting healthcare professionals in diagnosing and managing various pulmonary conditions with a synthesized understanding of both visual and textual data.

# Information
* **Base LLM** : LLAMA2-13B
* **Image Encoder** : Resnet50
* **Dataset** : Ensemble of multiple datasets including MIMIC. For more information, check our publication [pending]



# Examples
**Question**

    Carefully read the provided chest x-ray and write a detailed radiological report.

**LLM output**

    The chest radiograph shows significant opacification in the lower half of the right hemithorax, indicating a substantial pleural effusion. This could potentially obscure underlying lung pathology. There is noticeable displacement of the mediastinum to the left, suggesting possible mass effects. The left lung appears largely clear with no visible signs of pneumothorax.

**Question**

    What is possible diagnosis?

**LLM output**

    The diagnosis could be either pneumonia or atelectasis. 

**Question**

    Should additional radiologic study needed?

**LLM output**

    Given the presence of opacity and pleural effusion, further radiologic studies may be beneficial to monitor these conditions. A follow-up chest CT could provide more detailed information. 


# Requirement
## Hardware
You need a NVIDIA GPU with VRAM larger than 26GB. 
Recommended : NVIDIA-A100 40GB

## Software Dependency
torch, transformers, open_clip_torch
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





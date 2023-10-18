import os
from CXR_LLAVA.CXR_LLAVA import CXR_LLAVA_Loader
from PIL import Image
if __name__ == '__main__':
    model_path = "MODEL FOLDER PATH"
    img = Image.open(os.path.join(os.path.dirname(__file__), "IMG", "img.jpg"))
    loader = CXR_LLAVA_Loader(model_path=model_path, temperature=0.4, top_p=0.8)

    chat = [
        {"role": "system", "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
        {"role": "user", "content": "<image>\nWrite a radiologic report on the given chest radiograph, including information about atelectasis, cardiomegaly, consolidation, pulmonary edema, pleural effusion, and pneumothorax.\n"}
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




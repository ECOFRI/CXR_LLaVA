import os
from CXR_LLAMA.CXR_LLAMA import CXR_LLAMA_Loader
from PIL import Image
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_path = r"G:\Temp\finetune_result\LLAMA2-13B-CHAT_NewRN50_FINETUNE"
    loader = CXR_LLAMA_Loader(model_path=model_path, temperature=0, top_p=0.7)

    chat = [
        {"role": "system", "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
        {"role": "user", "content": "<image> Carefully read the provided chest x-ray and write a detailed radiological report."}
    ]
    img = Image.open(os.path.join(os.path.dirname(__file__),"IMG","consolidation.jpg"))
    response = loader.generate(chat,pil_image=img)
    print(response)






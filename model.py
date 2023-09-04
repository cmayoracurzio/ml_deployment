from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# Create model and input processor instances
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa')

# Model pipeline that is called in FastAPI server
def model_pipeline(text:str, image:Image):

    # Encode inputs
    encoding = processor(image,text,return_tensors="pt")

    # Inference 
    outputs = model(**encoding)
    logits = outputs.logits
    index = logits.argmax(-1).item()

    return model.config.id2label[index]
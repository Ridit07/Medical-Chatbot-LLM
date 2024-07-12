
from transformers import AutoTokenizer, AutoModelForCausalLM
import tensorflow as tf
from PIL import Image
import numpy as np

llama_model_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

dl_model_path = "/Users/riditjain/Downloads/Physician.Ai-main/Trained Model/skin cancer/skin_model.h5"
dl_model = tf.keras.models.load_model(dl_model_path)

def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))  
    image_array = np.array(image) / 255.0 
    return np.expand_dims(image_array, axis=0) 

def generate_text_with_image_analysis(image_path, input_text, tokenizer, llama_model, dl_model):
    image_input = preprocess_image(image_path)
    
    dl_output = dl_model.predict(image_input)
    
    predicted_class = np.argmax(dl_output, axis=1)[0]
    class_labels = {0: "possitive", 1: "negetive"} 
    detected_object = class_labels[predicted_class]
    
    adjusted_text = f"{input_text} The image contains {detected_object}."
    
    inputs = tokenizer.encode(adjusted_text, return_tensors="pt")
    outputs = llama_model.generate(inputs, max_length=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

image_path = "/Users/riditjain/Downloads/Physician.Ai-main/static/skin images/6ef5b8325ae265ec31dda54999657f60a9eba43b-341x359.webp"
input_text = "checl if its a skin cnacer"
output_text = generate_text_with_image_analysis(image_path, input_text, tokenizer, llama_model, dl_model)
print(output_text)

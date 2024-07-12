from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
import torch
from flask import Flask, redirect, request, render_template, url_for
import json
import os
import argparse
import socket
import base64
import io
from io import BytesIO
from PIL import Image
import time
import numpy as np

# model_id = "stabilityai/sdxl-turbo"
# pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = Flask(__name__)

# Main function to handle request
@app.route("/api", methods=['GET', 'POST'])
def handle_request():
    start_time = time.time()
    # ...
    prompt = request.form.get('prompt')
    
    # image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0] 
    image = pipe(prompt, generator=torch.manual_seed(0)).images[0]  
    image.save('sample.png')

    image = image.resize((336, 336))

    image_io = BytesIO()
    image.save(image_io, format='PNG')
    image_binary = image_io.getvalue()
    encoded_image = base64.b64encode(image_binary).decode('utf-8')
    
    process_time = time.time() - start_time
    print(f'Done processing in {process_time:0.4f} secs')

    return {
        'image': encoded_image
    }

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True)
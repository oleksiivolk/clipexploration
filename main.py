import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import torch
import clip
from PIL import Image
import numpy as np

from loader import get_nae_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("misc/CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

imgnet_a_loader = get_nae_loader()

with torch.no_grad():
    image_features = []
    
    # text_features = model.encode_text(text)

    for data, target in imgnet_a_loader:
        data, target = data.cuda(), target.cuda()

        image_features.append(model.encode_image(data).cpu().numpy())

        # accuracy
        # pred = output.data.max(1)[1]
        # num_correct += pred.eq(target.data).sum().item()

        # confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
        # pred = output.data.max(1)[1]
        # correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())
    # a = [print(x.shape) for x in image_features]
    encoded_imgs = np.concatenate(image_features, axis = 0)
    print(encoded_imgs.shape)
    
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
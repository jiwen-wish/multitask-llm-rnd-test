#%%
import torch
import clip
from PIL import Image
import requests
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default='data/wish_products_internallabel/wish_products_offshore_labelled_validated_processed.json')
parser.add_argument('--input_product_id_key', default='pid')
parser.add_argument('--output_file', default='data/wish_products_internallabel/wish_products_offshore_labelled_validated_processed_wclip.json')
args = parser.parse_args()

def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()

def clip_product(product_id = "591080f8dcb8722cca75999a"):

    image_data = load_image(f"https://contestimg.wish.com/api/webimage/{product_id}-large.jpg")
    image = preprocess(image_data).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_feature_list = image_features.detach().cpu().numpy()[0].tolist()
        return image_feature_list
# %%
if __name__ == "__main__":
    k = args.input_product_id_key
    with open(args.input_file, 'r') as fin, open(args.output_file, 'w') as fout:
        for l in tqdm(fin):
            if len(l):
                dat = json.loads(l)
                dat['img_embedding'] = clip_product(dat[k])
                fout.write(json.dumps(dat) + '\n')
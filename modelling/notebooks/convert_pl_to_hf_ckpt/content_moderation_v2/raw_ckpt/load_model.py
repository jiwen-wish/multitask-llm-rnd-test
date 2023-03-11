# Credits: Copied and modified from Qingqing Liu 
# URL: https://github.com/ContextLogic/knowledge-engineering-data/blob/78c17a2e9fd7bf542d58e22228ac5d03f3e9bd16/content_moderation/inappropriate_listing_v2_deploy/inference.py
# Slack: https://logicians.slack.com/archives/D043FKSMVLL/p1678482284782899

import os 
import torch
from tqdm.notebook import tqdm
import boto3
import pandas as pd
import numpy as np
import botocore.exceptions
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image
from catboost import CatBoostClassifier, Pool
import clip
CLIP_MODEL = "ViT-L/14"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MINIBATCH_SIZE = 32
DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 1

MODEL_FILEPATH = os.path.join(os.path.dirname(__file__), 'catboost_emb_noreko.cb')
TEXT_EMBEDDINGS_FILEPATH = os.path.join(os.path.dirname(__file__), 'text_embeddings.npy')

CATBOOST_MODEL = CatBoostClassifier().load_model(fname=MODEL_FILEPATH)
TEXT_EMBEDDINGS = np.load(TEXT_EMBEDDINGS_FILEPATH)

CATBOOST_MODEL_THRESHOLD = 0.9943

clip_model, clip_preprocess = clip.load(
    CLIP_MODEL,
    device=DEVICE
)

class ImageCLIP(torch.nn.Module):
    def __init__(self, model) :
        super().__init__()
        self.model = model
        
    def forward(self, image):
        return self.model.encode_image(image)
    
clip_model = torch.nn.DataParallel(
    ImageCLIP(clip_model),
)

class ImagesDataset(Dataset):
    def __init__(self, image_hash_list, transform=None):
        self._image_hashes = image_hash_list
        assert type(self._image_hashes) == list
        self._image_hashes = [h for h in self._image_hashes if h is not None]

        self._transform = transform

        self._s3_buckets = [
            "feed-processing-tmp-product-image-prod", # images in feed-processing
            "sweeper-production-productimage", # legacy images and online listings
        ]
        self._s3_client = None

    def __len__(self):
        return len(self._image_hashes)

    def __getitem__(self, idx):
        if type(idx) == slice:
            raise NotImplementedError("Cannot get more than one item")

        image_hash = self._image_hashes[idx]
        image_name = f"{image_hash}.jpg"

        resp = self._get_s3_object(image_name)
        if resp is None:
            print(f"Not found: {image_name}")
            return None, image_hash

        try:
            image = Image.open(resp['Body']).convert("RGB")
        except:
            print(f"Invalid Image: {image_name}")
            return None, image_hash

        if self._transform is not None:
            image = self._transform(image)

        return image, image_hash

    def _get_s3_object(self, object_key, buckets=None):
        s3_client = self._get_s3_client()
        buckets = self._s3_buckets if buckets is None else buckets

        resp = None
        for bucket in buckets:
            try:
                resp = s3_client.get_object(Bucket=bucket, Key=object_key)
            except botocore.exceptions.ClientError:
                continue
            else:
                break
        return resp

    def _get_s3_client(self):
        if self._s3_client is None:
            # create outside of init so 
            #print(f"Loading s3_client. In process: {torch.utils.data.get_worker_info()}")
            self._s3_client = boto3.client('s3')

        return self._s3_client


def my_collate_fn(batch):
    # remove any invalid images
    batch = [x for x in batch if x[0] is not None]
    return default_collate(batch)


def generate_dict_image_hash_to_feature(image_dataloader):
    for i, (image_t, image_hashes) in enumerate(tqdm(image_dataloader)):
        with torch.no_grad():
            image_t = image_t.to(DEVICE)
            image_features = clip_model.forward(image_t)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy() 

    return {x: y for (x, y) in zip(image_hashes, image_features)}

def mean_and_norm(nd_array):
    vector = np.mean(nd_array, axis=0)
    return vector/np.linalg.norm(vector)

def generate_image_embeddings(row, dict_image_hash_to_feature):
    image_embeddings = np.array([dict_image_hash_to_feature.get(x, np.zeros(768)) for x in row['images']]) 
    return mean_and_norm(image_embeddings)     

def generate_image_text_similarity(row):
    return np.dot(TEXT_EMBEDDINGS, row['image_embedding'].reshape((-1,1))).flatten()

def inference(requests):
    df = pd.DataFrame(requests['data'])

    all_image_hashes = []
    for _, row in df.iterrows():
        all_image_hashes.extend(row['images'])


    image_dataset = ImagesDataset(
        all_image_hashes,
        #transform=ToTensor(),
        transform=clip_preprocess,
    )

    image_dataloader = DataLoader(
        image_dataset,
        batch_size=MINIBATCH_SIZE * DEVICE_COUNT,
        num_workers=DEVICE_COUNT * 2,
        pin_memory=True,
        #shuffle=True,
        collate_fn=my_collate_fn,
    )

    dict_image_hash_to_feature = generate_dict_image_hash_to_feature(image_dataloader)    

    df['image_embedding'] = df.apply(
        lambda row: generate_image_embeddings(row, dict_image_hash_to_feature), 
        axis = 1 
    )   

    df['image_text_similarity'] = df.apply(
        generate_image_text_similarity, 
        axis = 1 
    ) 

    X = pd.DataFrame(df['image_text_similarity'].tolist(), columns=CATBOOST_MODEL.feature_names_[:-3])
    X['product_name'] = df['title']
    X['product_description'] = df['description']
    X['image_embedding'] = df['image_embedding']

    pool = Pool(
        X[CATBOOST_MODEL.feature_names_],
        cat_features=[],
        text_features=["product_name", "product_description"],
        embedding_features=["image_embedding"] if "image_embedding" in CATBOOST_MODEL.feature_names_ else None,
    )
    y_pred_proba = CATBOOST_MODEL.predict_proba(pool)   

    res = {}
    res['trace_id'] = requests['trace_id']
    response_list = list(1 * (y_pred_proba[:,0] < CATBOOST_MODEL_THRESHOLD))
    res['result'] = [{'response': x, 'error': '', 'other': ''} for x in response_list]
    res['other'] = requests['other']
    return res 


if __name__ == "__main__":
    # this is to mimic the input in https://docs.google.com/document/d/1sHxWGfDoS7IznYFCR6ye0bg6N2GiJ2ahIi3JqxI3MmA/edit#heading=h.s8q4io3mcaqd 
    requests = {
        'trace_id': 'trace_id666', 
        'data': 
            [
                {
                    'product_id': '63d1a67398b6bb61948e7441',
                    'title': 'Convenient Practical Multi-functional Gift Pouch Drawstring Auspicious Cloud Cloth Bracelet Bag Multi Color Jewelry Case Women Jewelry Bag Chinese Style Storage Bag Jewelry Organizer',
                    'description': 'Size: 11 × 11 cm\r\nMaterials: cloth\r\nColor: red, yellow, blue, green, pink, purple, etc\r\nPackage: 1 PC jewelry storage bag\nProduct description\r\nThis jewelry bag is small and exquisite, which is convenient to carry.\r\nNote: The size and weight are measured by hand, it is normal to have errors, please refer to the actual product received.',
                    'images': 
                        ['236005ff24e5b8b2b47896be92581ea7',
                        'ceb13f0ae37dcc260156d50265ce6a93',
                        'efa888054070eff0456dc0cb0b79e44b',
                        '1c1b8a696d15aa0c3b4c359f3b344276',
                        '4efae9d26fb00e0e55cd72e7f681f475',
                        '6051609aa6a1f46073fa5cf3e1035bf4',
                        'e62f5e0d748f2b6d886acf37f0362f7a',
                        '970fbbf9c24372dc64c5a2bbe939483c',
                        'b064bc1888b35c69b5cd1fc0abd0be8d',
                        '9a46463a732eb538140c3de49541f19e'], 
                    'price': None, 
                    'main_image_index': None
                }, 
                {
                    'product_id': '62cde59200713e1ee149400f',
                    'title': 'Ouch! Silicone Strapless Strapon - Black AND Dragon Alkaline AAA Batteries great gift',
                    'description': 'This sensational silicone strap on dildo requires no harness so nothing comes between you and your lover with the exception of your own intense pleasure together. Completely strapless it stays in place with an insertable ribbed curved probe delivering delicious . stimulation to the wearer. The longer ribbed end is used to penetrate your male or female partner. Ideal for same-sex couple or couples who are into pegging! Made from super-flexible Thermoplastic Rubber with a smooth medical-grade silicone finish this sexy toy has a curved . stimulation part of 6.8cm (2.68inch) while the penetrating end measures a full 12cm. Weight package 10.51 oz Product dimensions 4.02" x 1.61" x 1.61" Product weight 6.53 oz Product diameter 0.98" Insertable length 4.72" Waterproof Yes Splashproof Yes Phthalate free Yes Specifications . stimulation part: Length: 10 cm Diameter: 2 cm Penetrating part: Length 12 cm Diameter: 25 cm Materials Silicone TPR AND Dragon Alkaline AAA Batteries (not all items need batteries,  just nice to have around',
                    'images': 
                        ['8c6ca8dd51d875739fdf8ca2810ded58',
                        '8c6ca8dd51d875739fdf8ca2810ded58',
                        '8c6ca8dd51d875739fdf8ca2810ded58'], 
                    'price': None, 
                    'main_image_index': None    
                }
            ],
        'other': None
    }
    resp = inference(requests)
    print(resp)
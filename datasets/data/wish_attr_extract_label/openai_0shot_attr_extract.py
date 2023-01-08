#%%
import json
import pandas as pd
from tqdm import tqdm
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import time
llm = OpenAI(model_name='text-davinci-003', temperature=0, max_tokens=468)

# Note: just run this file from its local dir

# path can be changed
attribute_ontology = pd.read_csv('[ontology] wish_top25L2_attributes - 20221219.csv')
# shuffle to get diverse data in real time
example_products = pd.read_csv('SDT-871 top25 L2 product text_ 10K product listings - 2023-01-06.csv').sample(
    frac=1., random_state=42)

# setup global vars
l2set = set(attribute_ontology['wish_L2'])
df_tax = pd.read_json('../taxonomy/wish_newtax.json', lines=True)
taxid2path = {}
for i in df_tax.to_dict('records'):
    if len(i['category_path']) > 0:
        taxid2path[i['id']] = i['category_path']

# setup prompts
template = """Sentence: \"\"\"product title: {product_title}
product description: {product_description}
\"\"\"
Instruction: given the above product information which belongs to {taxonomy}, please extract entities and their types from the input sentence, all entity types are in options

Options: {attribute_types}

Entities (only for options specified above, formatted as json that can be parsed, ordered in the same way):
"""

prompt_product = PromptTemplate(
    input_variables=["product_title", "product_description", "taxonomy", "attribute_types"],
    template=template,
)

template = """{previous_text}

Now normalize above extracted enties, given the following specification that contains a list of of recommended normalized values for each entity type. If possible, please choose from them, but in rare cases where you have to, you can create new normalized value following similar style and semantics:

Specification:

{specification_json}

Normalized Entities:
"""

prompt_normalize = PromptTemplate(
    input_variables=["previous_text", "specification_json"],
    template=template,
)

# setup llm chains
def zero_shot_attribute_extraction_product_helper(product_title, product_description, taxonomy, l2):
    # prepare inputs
    attribute_ontology_l2 = attribute_ontology[attribute_ontology['wish_L2'] == l2]
    assert len(attribute_ontology_l2) > 0
    attribute_types_list = attribute_ontology_l2['attribute_name'].tolist()
    attribute_types = ", ".join(attribute_types_list)
    

    prompt_product_text = prompt_product.format(
        product_title=product_title, 
        product_description=product_description,
        taxonomy=taxonomy,
        attribute_types=attribute_types
    )

    all_text = product_title + "\n" + product_description

    product_attr_extract_json = llm(prompt_product_text)
    product_attr_extract_dict = json.loads(product_attr_extract_json)
    product_attr_extract_dict_clean = {}
    for i in product_attr_extract_dict:
        if i in attribute_types_list and product_attr_extract_dict[i] is not None:
            if isinstance(product_attr_extract_dict[i], str) and len(product_attr_extract_dict[i]) > 0 and \
                    product_attr_extract_dict[i].lower() in all_text.lower():
                product_attr_extract_dict_clean[i] = product_attr_extract_dict[i]
            elif isinstance(product_attr_extract_dict[i], list):
                if len(product_attr_extract_dict[i]) > 0:
                    tmp = []
                    for j in product_attr_extract_dict[i]:
                        if j is not None:
                            if len(j) > 0 and j.lower() in all_text.lower():
                                tmp.append(j)
                    if len(tmp) > 0:
                        product_attr_extract_dict_clean[i] = tmp
    product_attr_extract_json_clean = json.dumps(product_attr_extract_dict_clean, indent=2)

    # second call: normalize them
    specification = {} 
    for i in attribute_ontology_l2.to_dict('records'):
        if i['attribute_name'] in product_attr_extract_dict_clean:
            specification[i['attribute_name']] = i['example_attribute_value']
    specification_json = json.dumps(specification, indent=2)
    prompt_normalize_text = prompt_normalize.format(
        previous_text=prompt_product_text + '\n' + product_attr_extract_json_clean,
        specification_json=specification_json
    )
    product_normalize_json = llm(prompt_normalize_text)
    product_attr_extract_dict_clean_normalized_clean = {}
    product_attr_extract_dict_clean_normalized = json.loads(product_normalize_json)
    for k in product_attr_extract_dict_clean_normalized:
        v = product_attr_extract_dict_clean_normalized[k]
        existing_normalized_vals = attribute_ontology.loc[
            (attribute_ontology['wish_L2'] == l2) & (attribute_ontology['attribute_name'] == k), 
            'example_attribute_value'
        ].apply(lambda x: eval(x)).tolist()[0]
        if isinstance(v, str) and \
                len(v) > 0:
            if v.lower() in [i.lower() for i in existing_normalized_vals]:
                product_attr_extract_dict_clean_normalized_clean[k] = v
            else:
                product_attr_extract_dict_clean_normalized_clean[k] = v
                # Note: turns out doing this does more harm since normalized value gets large pretty fast
                # # update ontology with newly discovered normalized values
                # attribute_ontology.loc[
                #     (attribute_ontology['wish_L2'] == l2) & (attribute_ontology['attribute_name'] == k), 
                #     'example_attribute_value'
                # ] = str(existing_normalized_vals + [v])
        elif isinstance(product_attr_extract_dict_clean_normalized[k], list) and \
                len(product_attr_extract_dict_clean_normalized[k]) > 0:
            tmp = []
            for vi in v:
                existing_normalized_vals = attribute_ontology.loc[
                    (attribute_ontology['wish_L2'] == l2) & (attribute_ontology['attribute_name'] == k), 
                    'example_attribute_value'
                ].apply(lambda x: eval(x)).tolist()[0]
                if vi is not None and len(vi) > 0:
                    if vi.lower() in [i.lower() for i in existing_normalized_vals]:
                        tmp.append(vi)
                    else:
                        tmp.append(vi)
                        # Note: turns out doing this does more harm since normalized value gets large pretty fast
                        # # update ontology with newly discovered normalized values
                        # attribute_ontology.loc[
                        #     (attribute_ontology['wish_L2'] == l2) & (attribute_ontology['attribute_name'] == k), 
                        #     'example_attribute_value'
                        # ] = str(existing_normalized_vals + [vi])

    return product_attr_extract_dict_clean, product_attr_extract_dict_clean_normalized

def zero_shot_attribute_extraction_product(product_dict):
    return zero_shot_attribute_extraction_product_helper(
        product_title=product_dict["name"] ,
        product_description=product_dict["product_description"], 
        taxonomy=product_dict["category_path"], 
        l2=product_dict['L2']
    )

# %%
existing_pids = set(pd.read_json('size_10k_25l2_openai_0shot_pseudolabel.json', lines=True)['product_id'].tolist())
example_products_left = example_products[example_products.product_id.apply(lambda x: x not in existing_pids)]
print(f"{len(example_products_left)} out of {len(example_products)} products left TODO")
with open('size_10k_25l2_openai_0shot_pseudolabel.json', 'a', buffering=1) as f:
    for product_dict in tqdm(example_products_left.to_dict('records')):
        for _ in range(1):
            try:
                attr, normalized_attr = zero_shot_attribute_extraction_product(product_dict)
                product_dict['openai0shot_attr'] = attr 
                product_dict['openai0shot_attr_normalized'] = normalized_attr
                f.write(json.dumps(product_dict) + '\n')
                break
            except Exception as e:
                print(e)
                if _ != 0:
                    print('Try again after wait 1s')
                else:
                    print('Skip')
                time.sleep(1)

    #     # skip this since updating ontology in real time does more harm
    #     cnt += 1
    #     if cnt % 10 == 1:
    #         attribute_ontology.to_csv('[ontology] wish_top25L2_attributes_added_new_normalized_vals.csv', index=False)
    # attribute_ontology.to_csv('[ontology] wish_top25L2_attributes_added_new_normalized_vals.csv', index=False)
from locust import HttpUser, task, between
import numpy as np

class MyUser(HttpUser):
    wait_time = between(1, 2)
    host = 'http://localhost:8000'
    
    @task
    def classify_query(self):
        # q = (np.random.randint(1, 6) * 'dildo ').strip()
        self.client.post("/v2/models/query_classify_onnx/versions/1/infer", json={
        "inputs":[
            {	
                "name": "attention_mask",
                "shape": [1, 20],
                "datatype": "INT64",
                "data": [[1, 1, 1, 1] * 5]
            },
            {	
                "name": "input_ids",
                "shape": [1, 20],
                "datatype": "INT64",
                "data": [[1, 1, 1, 1] * 5]
            }
        ]
        })

if __name__ == "__main__":
    import os
    os.system("locust --headless -f locust_test.py -u 400 -r 10 -t 2m")
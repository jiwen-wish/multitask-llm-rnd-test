from locust import HttpUser, task, between
import numpy as np

class MyUser(HttpUser):
    wait_time = between(1, 2)
    host = 'http://localhost:8000'
    
    @task
    def classify_query(self):
        self.client.post("/v1/models/mymodel/predict", json={
            "id": "string",
            "parameters": {},
            "data": "string"
        })

if __name__ == "__main__":
    import os
    os.system("locust -f locast_test2.py -u 40 -r 10 -t 2m")
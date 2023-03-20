from locust import HttpUser, task, between
import numpy as np

class MyUser(HttpUser):
    wait_time = between(1, 2)
    host = 'http://coeus-gpu-multitask-ml-stage.service.consul:8081'
    
    @task
    def classify_query(self):
        q = (np.random.randint(1, 6) * 'dildo ').strip()
        self.client.post("/query-classify/v3", json={"text": f"Classify query: {q}"})

if __name__ == "__main__":
    import os
    os.system("locust --headless -f locast_test_coeus_multitask.py -u 40 -r 10 -t 2m")
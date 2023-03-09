from locust import HttpUser, task, between
import numpy as np
import json

class MyUser(HttpUser):
    wait_time = between(1, 2)
    host = 'http://localhost:8000'
    
    @task
    def classify_query(self):
        q = (np.random.randint(1, 6) * 'dildo ').strip()
        response = self.client.post("/", json={"text": f"Classify query: {q}"})
        # print(response.text)

if __name__ == "__main__":
    import os
    os.system("locust --headless -f locast_test_queryclassifyv3mini_rayserve.py -u 400 -r 10 -t 2m")
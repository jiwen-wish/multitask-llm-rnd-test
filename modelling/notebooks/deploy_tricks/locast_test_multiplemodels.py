from locust import HttpUser, task, between
import numpy as np

class MyUser(HttpUser):
    wait_time = between(1, 2)
    host = 'http://localhost:8000'
    
    @task
    def classify_query(self):
        q = (np.random.randint(1, 6) * 'dildo ').strip()
        self.client.post("/query", json={"text": f"Classify query: {q}"})
    
    @task
    def classify_sent(self):
        q = (np.random.randint(1, 6) * 'dildo ').strip()
        self.client.post("/sent", json={"text": f"Classify query: {q}"})

if __name__ == "__main__":
    import os
    os.system("locust --headless -f locast_test_multiplemodels.py -u 400 -r 10 -t 2m")
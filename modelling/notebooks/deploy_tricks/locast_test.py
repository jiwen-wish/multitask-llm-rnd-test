from locust import HttpUser, task, between
import numpy as np

class MyUser(HttpUser):
    wait_time = between(1, 2)
    host = 'http://localhost:8000'
    
    @task
    def classify_query(self):
        q = (np.random.randint(1, 6) * 'dildo ').strip()
        self.client.post("/", data=f"Classify query: {q}")

if __name__ == "__main__":
    import os
    os.system("locust -f locast_test.py -u 400 -r 10 -t 2m")
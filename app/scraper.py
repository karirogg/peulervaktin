from selenium import webdriver 
from selenium.webdriver.chrome.service import Service as ChromeService 
from webdriver_manager.chrome import ChromeDriverManager 
from selenium.webdriver.common.by import By
import time
from selenium import webdriver
from PIL import Image
import base64
import cv2
import numpy as np
import random

from prediction import predict

from dotenv import load_dotenv
load_dotenv()
import os
import MySQLdb

connection = MySQLdb.connect(
  host= os.getenv("HOST"),
  user=os.getenv("USERNAME"),
  passwd= os.getenv("PASSWORD"),
  db= os.getenv("DATABASE"),
#  ssl_mode = "VERIFY_IDENTITY",
#  ssl      = {
#    "ca": "/etc/ssl/cert.pem"
#  }
)

cursor = connection.cursor()

url = "https://projecteuler.net/sign_in" 
 
options = webdriver.ChromeOptions() 
options.add_argument('--headless=new')

def get_updates():
    with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) as driver: 	
        driver.get(url)

        cursor.execute('SELECT MAX(id) FROM Classifications')
        max_id = cursor.fetchall()[0][0]

        while True:
            # max_id += 1

            element = driver.find_elements(By.XPATH,'//img[@id="captcha_image"]')[0]

            with open('../store/img.png', 'wb') as f:
                f.write(element.screenshot_as_png)

            im = cv2.imread('../store/img.png')

            im = im[1:69, 1:137]
            im[-1,:,:] = 255

            cv2.imwrite('../store/img2.png', im)

            with open(f"../store/img2.png", "rb") as image_file:
                base64str = base64.b64encode(image_file.read()).decode("utf-8")

            import requests, json
            payload = json.dumps({
                "base64str": base64str,
            })

            # response = requests.post("https://karirogg--peulbot-predict.modal.run",data = payload)
            # data_dict = response.json()

            captcha_prediction, probs = predict(base64str)

            driver.find_element(By.XPATH,'//input[@id="username"]').send_keys(os.getenv("PEULER_USERNAME"))
            driver.find_element(By.XPATH,'//input[@id="password"]').send_keys(os.getenv("PEULER_PASSWORD"))
            driver.find_element(By.XPATH,'//input[@id="captcha"]').send_keys(captcha_prediction)

            # time.sleep(5)

            driver.find_element("name", "sign_in").click()

            warnings = driver.find_elements(By.XPATH, '//p[@class="warning"]')

            new_id = random.randint(0, 1000000000)

            correct = 0
            if len(warnings) == 0:
                # cursor.execute('INSERT INTO CorrectlyClassified (img, prediction) VALUES (%s, %s)', (base64str, captcha_prediction))
                cv2.imwrite(f'../correct/{new_id}.png', im)
                correct = 1
            else:
                cv2.imwrite(f'../incorrect/{new_id}.png', im)

            cv2.imwrite(f'../images/{new_id}.png', im)

            time.sleep(2)

            cursor.executemany('INSERT INTO Classifications (id, prediction, number, digit, probability, correct) VALUES (%s, %s, %s, %s, %s, %s)', [(new_id, captcha_prediction, int(0), int(j), round(float(probs[0][j]), 4), correct) for j in range(10)])
            cursor.executemany('INSERT INTO Classifications (id, prediction, number, digit, probability, correct) VALUES (%s, %s, %s, %s, %s, %s)', [(new_id, captcha_prediction, int(1), int(j), round(float(probs[1][j]), 4), correct) for j in range(10)])
            cursor.executemany('INSERT INTO Classifications (id, prediction, number, digit, probability, correct) VALUES (%s, %s, %s, %s, %s, %s)', [(new_id, captcha_prediction, int(2), int(j), round(float(probs[2][j]), 4), correct) for j in range(10)])
            cursor.executemany('INSERT INTO Classifications (id, prediction, number, digit, probability, correct) VALUES (%s, %s, %s, %s, %s, %s)', [(new_id, captcha_prediction, int(3), int(j), round(float(probs[3][j]), 4), correct) for j in range(10)])
            cursor.executemany('INSERT INTO Classifications (id, prediction, number, digit, probability, correct) VALUES (%s, %s, %s, %s, %s, %s)', [(new_id, captcha_prediction, int(4), int(j), round(float(probs[4][j]), 4), correct) for j in range(10)])

            if correct == 1:
                break


        driver.find_element(By.XPATH, '//a[@href="friends"]').click()

        link_element_list = driver.find_elements(By.XPATH, '//table[@id="friends_table"]/tbody/tr/td[@class="username_column"]/table/tbody/tr/td/div/a')

        link_list = []

        for link in link_element_list:
            link_list.append(link.get_attribute("href"))

        out = []

        cursor.execute('SELECT username, problem FROM problems')

        result = cursor.fetchall()

        for link in link_list:
            driver.get(link)

            time.sleep(0.5)

            username = driver.find_element(By.XPATH, '//h2[@id="profile_name_text"]').text

            problems = driver.find_elements(By.XPATH, '//td[@class="tooltip problem_solved"]/a/div')

            if username == "peulervaktin":
                problems = driver.find_elements(By.XPATH, '//td[@class="tooltip problem_solved"]/a')

            solved_problems = []

            for problem in problems:
                solved_problems.append(int(problem.text))

            # Comparing real solved with database

            db_solved = [problem for (u, problem) in result if u == username]

            new_solved = list(set(solved_problems) - set(db_solved))

            cursor.executemany('INSERT INTO problems (username, problem) VALUES (%s, %s)', [(username, problem) for problem in new_solved])

            if len(new_solved) > 0:
                greeting = f"{username} var að leysa dæmi {new_solved[0]}! 🫡🐐"

                if len(new_solved) > 1:
                    greeting = f"{username} var að leysa dæmi {', '.join(np.array(new_solved[:-1], dtype=str))} og {new_solved[-1]}! 🫡🐐"

                out.append(greeting)
        
    connection.commit()

    return out

# 93837
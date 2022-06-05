from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os
from selenium.webdriver.common.by import By

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def crawling_img(name):
    # options=webdriver.ChromeOptions()
    # options.add_experimental_option("excludeSwitches", ["enable-logging"])
    # driver= webdriver.Chrome(options=options)
    driver= webdriver.Chrome()
    driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
    elem=driver.find_element(By.NAME, "q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)

    SCROLL_PAUSE_TIME =1
    last_height=driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height =driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element(by=By.CSS_SELECTOR, value=".mye4qd").click()
            except:
                break
        last_height=new_height

    imgs = driver.find_elements(by=By.CSS_SELECTOR, value=".rg_i.Q4LuWd")
    dir="./input/"+name
    createDirectory(dir)
    count=1
    for img in imgs:
        try:
            img.click()
            time.sleep(2)
            imgUrl=driver.find_element(by=By.XPATH,value='//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img').get_attribute("src")
            path="C:/Users/user/Desktop/crolling/input/"+name+"/"
            urllib.request.urlretrieve(imgUrl, path+name+str(count)+".jpg")
            count=count+1
            if count>=5000:
                break
        except:
            pass
    driver.close()
words=["passe","arabesque","plie"]

for word in words:
    crawling_img(word)

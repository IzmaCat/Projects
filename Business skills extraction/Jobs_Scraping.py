# -*- coding: utf-8 -*-

from selenium import webdriver
import pandas as pd
#from bs4 import BeautifulSoup
import time
from selenium.webdriver.support.wait import WebDriverWait
import json
        
driver=webdriver.Chrome(executable_path=r"C:/Users/*/anaconda3/Scripts/chromedriver.exe")        
        
#driver = webdriver.Firefox(executable_path=r"C:/Users/*/anaconda3/Scripts/geckodriver.exe")
#driver.get("https://www.indeed.com/jobs?q=diversity+and+inclusion+AND+people+analytics&start=" + str(10*1))
#time.sleep(2)
wait = WebDriverWait(driver, 60)
dataframe = pd.DataFrame(columns=["Title", "Company", "Description"])
for page in range( 140,145 ):
    #+ str(page))
    #driver.get("https://www.indeed.com/jobs?q=diversity+and+inclusion+AND+people+analytics&start="   + str(10*page))
    driver.get("https://www.indeed.com/jobs?q=diversity+and+inclusion&explvl=mid_level&start=" +str(10*page))
    time.sleep(2)
    summaryItems = driver.find_elements_by_xpath("//a[contains(@class, 'jobtitle turnstileLink')]")
    job_links = [summaryItem.get_attribute("href") for summaryItem in summaryItems]

    for job_link in job_links:
        driver.get(job_link)
        time.sleep(1)
       # frame = driver.find_element_by_xpath('//iframe[contains(@class, "h-captcha")]')
       # driver.switch_to.frame(frame)
       # driver.find_element_by_xpath("//*[@id='invisible-hcaptcha-div']")
        try:
            title = driver.find_element_by_xpath("//*[@class='icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title']").text
        except:
            title="None"
        try:
         company = driver.find_element_by_xpath("//*[@class='icl-u-lg-mr--sm icl-u-xs-mr--xs']").text
        except:
            company="None"
        try:    
        # location = driver.find_element_by_xpath("//*[@class='jobsearch-JobMetadataHeader-iconLabel']").text
            jd = driver.find_element_by_xpath("//*[@class='jobsearch-jobDescriptionText']").text
        except:
            jd="None"
            
        dataframe = dataframe.append({'Title': title,
                                          
                                          "Company": company,
                                          
                                          "Description": jd},
                                         ignore_index=True)
print(dataframe)

dataframe.to_csv('mid-d+i-140-145.csv', index = False)


















     

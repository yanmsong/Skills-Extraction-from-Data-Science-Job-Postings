import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager



def extract_job_title(soup): 
    jobs = []
    for div in soup.find_all(name="div", attrs={"class":"title"}):
        try:
            job_title = div.find(name='a', attrs={"data-tn-element":"jobTitle"})['title']
        except:
            job_title = "None"
        jobs.append(job_title)
    return(jobs)


def extract_company(soup): 
    companies = []
    for div in soup.find_all(name="div", attrs={"class":"sjcl"}):
        try:
            company = div.find(name="span", attrs={"class":"company"}).text.strip()
        except:
            company = "None"
        companies.append(company)
    return(companies)


def extract_location(soup): 
    locations = []
    for div in soup.find_all(name="div", attrs={"class":"sjcl"}):
        try:
            # location under div tag
            loc = div.find(name="div", attrs={"class":"location accessible-contrast-color-location"}).text
        except: 
            loc = "None"
        if loc=="None":
            try:
                # location under span tag
                loc = div.find(name="span", attrs={"class":"location accessible-contrast-color-location"}).text
            except:
                loc = "None"
        locations.append(loc)
    return locations


def extract_salary(soup): 
    salaries = []
    for div in soup.find_all(name="div", attrs={"class":"row"}):
        try:
            salary = div.find(name="span", attrs={"class":"salaryText"}).text.strip()
        except:
            salary = "None"
        salaries.append(salary)
    return(salaries)


def scrap(URL):
	driver = webdriver.Chrome(ChromeDriverManager().install())

	df = pd.DataFrame(columns=["Job Title","Location","Company","Salary","Description"])

	# scrap up to page 100 (start=990)
	for i in range(0,1000,10):
	    
	    driver.get(URL+str(i))
	    driver.implicitly_wait(8)
	    
	    all_jobs = driver.find_elements_by_class_name('result')
	    
	    for job in all_jobs:
	        result_html = job.get_attribute('innerHTML')
	        soup = BeautifulSoup(result_html,'html.parser')
	        
	        ### Job Title
	        try:
	            job_title = soup.find(name='a', attrs={"data-tn-element":"jobTitle"})['title']
	        except:
	            job_title = "None"
	        
	        ### Company
	        try:
	            company = soup.find(name="span", attrs={"class":"company"}).text.strip()
	        except:
	            company = "None"
	            
	        ### Location
	        try:
	            # location under div tag
	            loc = soup.find(name="div", attrs={"class":"location accessible-contrast-color-location"}).text
	        except: 
	            loc = "None"
	        if loc == "None":
	            try:
	                # location under span tag
	                loc = soup.find(name="span", attrs={"class":"location accessible-contrast-color-location"}).text
	            except:
	                loc = "None"
	            
	        ### Salary
	        try:
	            salary = soup.find(name="span", attrs={"class":"salaryText"}).text.strip()
	        except:
	            salary = "None"
	        
	        ### Description
	        sum_div = job.find_elements_by_class_name("summary")[0]
	    
	        try:
	            # click to display detailed job description
	            sum_div.click()
	        except:
	            close_button = driver.find_elements_by_class_name("popover-x-button-close")[0]
	            close_button.click()
	            sum_div.click()

	        try:
	            desc = driver.find_element_by_id('vjs-desc').text
	        except:
	            desc = job.find_elements_by_class_name("summary")[0].text

	        df = df.append({'Job Title':job_title,'Company':company,'Location':loc,
	                        'Salary':salary,'Description':desc},ignore_index=True)

	return df



if __name__ == '__main__':

	### data scientist
	URL = "https://www.indeed.com/jobs?q=Data+Scientist&l=US&start="
	df = scrap(URL)
	df.to_csv('indeed-all-ds-0406.csv',index=False)

	### data analyst
	URL = "https://www.indeed.com/jobs?q=data+analyst&l=US&start="
	df = scrap(URL)
	df.to_csv('indeed-all-da-0406.csv',index=False)







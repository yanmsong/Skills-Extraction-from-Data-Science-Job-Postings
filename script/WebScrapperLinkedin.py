from lxml import html, etree
import requests
import re
import os
import sys
import unicodecsv as csv
import argparse
import json
import pandas as pd


def parse(response):
    '''parse a website for general job information (without description)'''
    parser = html.fromstring(response.text)
    XPATH_ALL_JOB = '//li[@class="result-card job-result-card result-card--with-hover-state"]'
    XPATH_NAME = './/span[@class="screen-reader-text"]/text()'
    XPATH_JOB_URL = './/a[@class="result-card__full-card-link"]/@href'
    XPATH_LOC = './/span[@class="job-result-card__location"]/text()'
    XPATH_COMPANY = './/a[@class="result-card__subtitle-link job-result-card__subtitle-link"]/text()'
    XPATH_JD = './/p[@class="job-result-card__snippet"]/text()'
    XPATH_NEXTPAGE = '//li[@class="next"]//a/@href'

    listings = parser.xpath(XPATH_ALL_JOB)
    job_listings = []
    
    for job in listings:
        
        raw_job_name = job.xpath(XPATH_NAME)
        raw_job_url = job.xpath(XPATH_JOB_URL)
        raw_lob_loc = job.xpath(XPATH_LOC)
        raw_company = job.xpath(XPATH_COMPANY)
        raw_jd = job.xpath(XPATH_JD)
        
        # Cleaning data
        job_name = ''.join(raw_job_name).strip('–') if raw_job_name else None
        job_location = ''.join(raw_lob_loc) if raw_lob_loc else None
        raw_state = re.findall(",\s?(.*)\s?", job_location)
        state = ''.join(raw_state).strip()
        raw_city = job_location.replace(state, '')
        city = raw_city.replace(',', '').strip()
        company = ''.join(raw_company).strip().replace('–','')
        job_url = raw_job_url[0] if raw_job_url else None
    
        jobs = {
                    "Name": job_name,
                    "Company": company,
                    "State": state,
                    "City": city,
                    "Location": job_location,
                    "Url": job_url
                }
        job_listings.append(jobs)
        
    return job_listings



def parse_job_description(url):
    '''parse the website for each job for job description'''
    response = requests.get(url)
    parser = html.fromstring(response.text)
    XPATH_JD = './/section[@class="description"]/descendant::*/text()'
    description = parser.xpath(XPATH_JD)
    return description


if __name__ == "__main__":
    filename = '../data/linkedin_data_analyst.csv'

    # scrapped the 1000 jobs from the website
    i = 0
    response = requests.get("https://www.linkedin.com/jobs/search?keywords=Data%20Analyst&location=%E7%BE%8E%E5%9B%BD&trk=public_jobs_jobs-search-bar_search-submit&redirect=false&position=1&pageNum=0")
    job_listings = parse(response)
    while True:
        if len(job_listings) >= 999:
            break
        try:
            start = len(job_listings)
            response = requests.get("https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?location=%E7%BE%8E%E5%9B%BD&redirect=false&keywords=Data%20Analyst&trk=homepage-jobseeker_recent-search&position=1&pageNum=0&start={}".format(start))
            job_listings += parse(response)
        except:
            break
            
    # write the scrapped data to csv file
    scraped_data = job_listings
    with open(filename, 'wb')as csvfile:
        fieldnames = ['Name', 'Company', 'State',
                      'City', 'Salary', 'Location', 'Url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,quoting=csv.QUOTE_ALL)
        writer.writeheader()
        if scraped_data:
            for data in scraped_data:
                writer.writerow(data)
        else:
            print("Your search for %s, in %s does not match any jobs"%(keyword,place))
            
    # read the file and scrap the job description from each website       
    df = pd.read_csv(filename)
    df['Description'] = df['Url'].apply(parse_job_description)
    df.to_csv(filename)
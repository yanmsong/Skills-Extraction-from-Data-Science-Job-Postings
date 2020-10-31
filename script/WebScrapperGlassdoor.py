from lxml import html, etree
import requests
import re
import os
import sys
import unicodecsv as csv
import argparse
import json
import pandas as pd


headers = {'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'accept-encoding': 'gzip, deflate, sdch, br',
                'accept-language': 'en-GB,en-US;q=0.8,en;q=0.6',
                'referer': 'https://www.glassdoor.com/',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/51.0.2704.79 Chrome/51.0.2704.79 Safari/537.36',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
}

location_headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.01',
    'accept-encoding': 'gzip, deflate, sdch, br',
    'accept-language': 'en-GB,en-US;q=0.8,en;q=0.6',
    'referer': 'https://www.glassdoor.com/',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/51.0.2704.79 Chrome/51.0.2704.79 Safari/537.36',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
}
    
def parse_recursively(keyword, place_id, url=None, job_listings=[]):
    try:
        if not url:
            data = {"term": place_id,
                "maxLocationsToReturn": 10}

            location_url = "https://www.glassdoor.co.in/findPopularLocationAjax.htm?"
            try:
                # Getting location id for search location
                print("Fetching location details")
                location_response = requests.post(location_url, headers=location_headers, data=data).json()
                place_id = location_response[0]['locationId']
                job_litsting_url = 'https://www.glassdoor.com/Job/jobs.htm'
                # Form data to get job results
                data = {
                    'clickSource': 'searchBtn',
                    'sc.keyword': keyword,
                    'locT': 'C',
                    'locId': place_id,
                    'jobType': ''
                }
                if place_id:
                    response = requests.post(job_litsting_url, headers=headers, data=data)
            except:
                print("Failed to load locations")
            response = requests.post(job_litsting_url, headers=headers, data=data)
        else:
            response = requests.post(url, headers=headers)

        parser = html.fromstring(response.text)

        # Making absolute url 
        base_url = "https://www.glassdoor.com"
        parser.make_links_absolute(base_url)

        XPATH_ALL_JOB = '//li[@class="jl"]'
        XPATH_NAME = './/a/text()'
        XPATH_JOB_URL = './/a/@href'
        XPATH_LOC = './/span[@class="subtle loc"]/text()'
        XPATH_COMPANY = './/div[@class="jobInfoItem jobEmpolyerName"]/text()'
        XPATH_SALARY = './/span[@class="salaryText"]/text()'
        XPATH_NEXTPAGE = '//li[@class="next"]//a/@href'

        listings = parser.xpath(XPATH_ALL_JOB)
        next_url = job.xpath(XPATH_NEXTPAGE)
        
        for job in listings:
            raw_job_name = job.xpath(XPATH_NAME)
            raw_job_url = job.xpath(XPATH_JOB_URL)
            raw_lob_loc = job.xpath(XPATH_LOC)
            raw_company = job.xpath(XPATH_COMPANY)
            raw_salary = job.xpath(XPATH_SALARY)

            # Cleaning data
            job_name = ''.join(raw_job_name).strip('–') if raw_job_name else None
            job_location = ''.join(raw_lob_loc) if raw_lob_loc else None
            raw_state = re.findall(",\s?(.*)\s?", job_location)
            state = ''.join(raw_state).strip()
            raw_city = job_location.replace(state, '')
            city = raw_city.replace(',', '').strip()
            company = ''.join(raw_company).strip().replace('–','')
            salary = ''.join(raw_salary).strip()
            job_url = raw_job_url[0] if raw_job_url else None
            

            jobs = {
                "Name": job_name,
                "Company": company,
                "State": state,
                "City": city,
                "Salary": salary,
                "Location": job_location,
                "Url": job_url
            }
                
            job_listings.append(jobs)
        
        print(len(job_listings))
        
        if next_url:
            next_url = next_url[0]
            job_listings = parse_recursively(keyword, place_id, next_url, job_listings)
        else:
            print("return", len(job_listings))
            return job_listings
        return job_listings
    
    except:
        print("except return", len(job_listings))
        return job_listings
    
    
    
def parse_job_description(url):
    # extract the detailed job description
    try:
        print(url)
        page= requests.get(url, headers=headers)
        print(page)
        parser = html.fromstring(page.text)
        XPATH_JD = ".//div[@class='desc css-58vpdc ecgq1xb3']/text()"
        raw_job_description = parser.xpath(XPATH_JD)
        job_description = ''.join(raw_job_description)
        print(len(job_description))
        return job_description
    except:
        return ""
    

def parse_rows(row):
    try:
        if len(row['Description']) == 0:
            row['Description'] = parse_job_description(row['Url'])
    except:
        try:
            row['Description'] = parse_job_description(row['Url'])
        except:
            row['Description'] = ''
    return row
    
    
if __name__ == '__main__':
    
    keywords = ["Data Analyst"]
    place_id = "United States"
    filename = '../data/glassdoor_data_analyst.csv'
    
    # scrape the first 30 pages of jobs
    scraped_data = []
    for keyword in keywords:
        result = parse_recursively(keyword, place_id, url=None, job_listings=[])
        print(keyword, len(result))
        scraped_data += result
    print("Writing data to output file")
    
    # write the first pages of jobs to csv file
    with open(filename, 'wb')as csvfile:
        fieldnames = ['Name', 'Company', 'State',
                      'City', 'Salary', 'Location', 'Url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,quoting=csv.QUOTE_ALL)
        writer.writeheader()
        if scraped_data:
            for data in scraped_data:
                writer.writerow(data)
        else:
            print("Your search for %s, in %s does not match any jobs"%(keyword,place_id))
    
    # scrape each job website for detailed description
    df = pd.read_csv(filename)
    df = df.apply(parse_rows, axis=1)
    df.to_csv(filename)
        

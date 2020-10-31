import pandas as pd
from cleantext import clean # pip install clean-text
import re
import string

def clean_description(text):
    text = str(text)
    text = re.sub("', '", '.', text)
    text = re.sub('", "', '.', text)
    text = re.sub("[\[\]\']", '', text)
    
    return text

def clean_name(text):
    '''clean job title'''
    text = text.lower()
    return text

def clean_company(text):
    '''clean company name'''
    text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace(' ', '')
    return text

def clean_salary_min(text):
    '''get minimum salary'''
    min_sal= ''
    if type(text) == str:
        text_ls = text
        match = re.match(r"\$(?P<min_sal>\d+)k-\$(?P<max_sal>\d+)k", text)
        if match:
            min_sal = int(match.group('min_sal'))
    return min_sal

def clean_salary_max(text):
    '''get maximum salary'''
    max_sal = ''
    if type(text) == str:
        text_ls = text
        match = re.match(r"\$(?P<min_sal>\d+)k-\$(?P<max_sal>\d+)k", text)
        if match:
            max_sal = int(match.group('max_sal'))
    return max_sal


statesdic = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'}

statesdic_reverse = {v:k for (k,v) in statesdic.items()}

def clean_location(text):
    try:
        text_split = text.split(', ')
        if len(text_split) > 2:
            # replace the state by the short form(this is for linkedin)
            text_split = text_split[:2]
            if len(text_split[1]) > 2:
                text_split[1] = statesdic_reverse[text_split[1]]
            return ', '.join(text_split)
        elif len(text_split[1]) > 2:
            # match the pattern (this is for indeed)
            match = re.match(r"(?P<state>\w+)", text_split[1])
            text_split[1] = match.group('state')
            return ', '.join(text_split)
        return text
    except:
        return text
    
def clean_text(text):
    '''
    cleaning the text for the part of speech tagging
    input a string, return a string
    '''
    text = text.lower()

    text = re.sub(r"[()<>/]", ', ', text) # sub ()<>&/ to comma and space
    text = re.sub(r"&", 'and', text) # sub ()<>&/ to comma and space
    text = re.sub(r"[?!]", '. ', text) # sub ?! to dot and space

    text = re.sub(" [a-z0-9]+[\.'\-a-z0-9_]*[a-z0-9]+@\w+\.com", "", text) # sub email address to dot
    text = re.sub('[#"\']', '', text) # remove '#'

    text = re.sub("e\.g\.", '', text) # remove the 'e.g.'
    text = re.sub("it’s", 'it is', text)
    text = re.sub("we’re", 'we are', text)
    text = re.sub("[\t\n\r\f\v]+", ". ", text) # remove \n and \r

    text = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text) # add space after comma and dot


    text = re.sub('\W+\.', '.', text) # remove the empty space before a dot
    text = re.sub('\W+\,', ',', text) # remove the empty space before a comma
    text = re.sub('[,\.]+\.+', '.', text) # sub multiple dots to one dot
    text = re.sub(' +',' ',text) # replace multiple whitespace by one whitespace

    return text

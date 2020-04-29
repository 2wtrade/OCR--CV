# -*- coding: utf-8 -*-
"""

"""
# Load required packages
import numpy as np
import pandas as pd
import os
import datetime

name_sex_df=pd.read_csv(os.getcwd()+"\\dictionaries\\names.csv") # for names
surname_df=pd.read_csv(os.getcwd()+"\\dictionaries\\surnames.csv") # for surnames
patronymic_sex_df=pd.read_csv(os.getcwd()+"\\dictionaries\\patronymics.csv") # for patronymics
city_df=pd.read_csv(os.getcwd()+"\\dictionaries\\cities.csv") # for cities
code_df=pd.read_csv(os.getcwd()+"\\dictionaries\\codes.csv") # for codes and place

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Functions
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# test string as integer object
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
# test string as date object
def is_date(s):
    try: 
        datetime.datetime.strptime(s, '%d.%m.%Y')
        return True
    except ValueError:
        return False
    
 
# Take text array and find a patronnymic and sex into it W
def get_patronymic_sex2(text_c):
    patronymic_V=np.array(patronymic_sex_df.Patronymic,dtype=object) # get all patronnymics    
    sex_V=np.array(patronymic_sex_df.Sex,dtype=object) # get all sexes
    patronymic=''
    sex=''
    p=0
    for i in range(len(patronymic_V)):
        str_i=patronymic_V[i].upper()
        str_i=str_i.replace("\'","")
        pos=text_c.find(str_i)
        if pos>=0:
            if len(str_i)>len(patronymic):
                patronymic=str_i
                sex=sex_V[i]
                p=pos
    return [patronymic,sex,p]

# Take text and find a number of a passport W
def get_series2(text_c):
    ser=''
    for i in range(len(text_c)-5):
        str_i=text_c[i:(i+6)]
        if is_int(str_i.replace(" ",""))==True and len(str_i.replace(" ",""))==4:
            if str_i[2]==' ':
                ser=str_i
                break   
            if str_i[0]==' ' and str_i[5]==' ':
                ser=str_i
                break
    return [ser]

# Take text and find a number of a passport W
def get_number2(text_c):
    number=''
    p=0
    for i in range(len(text_c)-5):
        str_i=text_c[i:(i+6)]
        if is_int(str_i)==True and len(str_i.replace(" ",""))==6:
            number=str_i
            p=i
            break
    return [number,p]

# Take text and find a code and place into it W
def get_code_place2(text_c):
    code_V=np.array(code_df.Code,dtype=object) # get codes
    place_V=np.array(code_df.Place,dtype=object) # get places
    code=''
    place=''
    p=0
    for i in range(len(code_V)):
        str_i=code_V[i].upper().replace("\'","")
        pos=text_c.find(str_i)
        if pos>=0:
            code=str_i
            place=place_V[i]
            p=pos
            break
    return [code,place,p]

# Take text and find all dates in it W
def get_dates2(text_c):
    birth_date=''
    issue_date=''
    for i in range(len(text_c)-9):
        d = text_c[i:(i+10)]
        if is_date(d)==True:
            if i/(1+len(text_c))<0.5:
                birth_date=d
            if i/(1+len(text_c))>0.5:
                issue_date=d                
    return [birth_date,issue_date]

# Take text and find a city into it W
def get_city2(text_c,bp=0):
    city_V=np.array(city_df.Cities,dtype=object) # get patronnymic
    city=''
    p=0
    if bp>0:
        text_c=text_c[(bp+1):len(text_c)]
    for i in range(len(city_V)):
        str_i=city_V[i].upper()
        str_i=str_i.replace("\'","")
        pos=text_c.find(str_i)
        if pos>=0:
            if p==0:
                city=str_i
                p=pos
                continue
            if pos>=p:
                city=str_i
                p=pos
    return [city,p+bp+1]

# Take text and find a name and sex into it W
def get_name2(text_c, sex_c, bp=0):
    name_V=np.array(name_sex_df.Name,dtype=object)
    if sex_c!='':
        name_V=np.array(name_sex_df.loc[name_sex_df.Sex==sex_c].Name,dtype=object) # get names
    name=''
    p=0
    if bp>0:
        text_c=text_c[0:bp]
    for i in range(len(name_V)):
        str_i=name_V[i].upper()
        str_i=str_i.replace("\'","")
        pos=text_c.find(str_i)
        if pos>=0:
            if p==0:
                name=str_i
                p=pos
                continue
            if (pos+len(str_i))>(p+len(name)):
                name=str_i
                p=pos
                continue
            if (pos+len(str_i))==(p+len(name)) and len(str_i)>=len(name):
                name=str_i
                p=pos
    return [name,p]

# Take text and find a surname into it W
def get_surname2(text_c,bp=0):
    surname_V=np.array(surname_df.Surname,dtype=object) # convert to upper case # get names
    surname=''
    p=0
    if bp>0:
        text_c=text_c[0:bp]
    for i in range(len(surname_V)):
        str_i=surname_V[i].upper()
        str_i=str_i.replace("\'","")
        pos=text_c.find(str_i)
        if pos>=0:
            if p==0:
                surname=str_i
                p=pos
                continue
            if (pos+len(str_i))>(p+len(surname)):
                surname=str_i
                p=pos
                continue
            if (pos+len(str_i))==(p+len(surname)) and len(str_i)>=len(surname):
                surname=str_i
                p=pos
    return [surname,p]

# Text parsing
def parse_text2(text_c0,text_c90):    
    # Get patronymic and sex
    patronymic_sex=get_patronymic_sex2(text_c0) # get patronymic
    # Get city
    p=patronymic_sex[2]
    if p==0:
        p=int(len(text_c0)/2)
    city=get_city2(text_c0, p)
    # Get name
    if p==0:
        p=int(len(text_c0)/2)
    name=get_name2(text_c0, patronymic_sex[1],p)
    # Get surname
    p=name[1]
    if p==0:
        p=int(len(text_c0)/2)
    surname=get_surname2(text_c0, p)
    # Get dates
    dates=get_dates2(text_c0)
    # Get code
    code_place=get_code_place2(text_c0)
    # Get number of passport
    number=get_number2(text_c90)
    # Get series of passport
    series=get_series2(text_c90)
    return pd.DataFrame({'Дата выдачи':[dates[1]],'Код подразделения':[code_place[0]],
                         'Кем выдан':[code_place[1]],'Серия':[series[0]], 'Номер':[number[0]],
                         'Фамилия':[surname[0]],'Имя':[name[0]],
                         'Отчество':[patronymic_sex[0]], 'Пол':[patronymic_sex[1]],
                         'Год рождения':[dates[0]],'Город рождения':[city[0]]})
    

    

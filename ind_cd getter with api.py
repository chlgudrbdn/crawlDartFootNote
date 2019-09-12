
from selenium import webdriver
import time
import pandas as pd
from urllib.parse import urljoin
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support import expected_conditions

# from timedelta import datetime

import requests
import json
from konlpy.utils import pprint

import requests
import lxml.html
import re
apikey = '30f25809837369a4c4aaf5046a55e002b38f8ce0'
# type(apikey), len(apikey)
url_company = "http://dart.fss.or.kr/api/company.json?auth={0}&crp_cd={1}"

crp_cd_df = pd.read_excel('crp_cd list.xlsx', dtype=str)  # 단순히 KRX에서 상장 회사 리스트를 다운받기만 하면 상장 폐지된 회사는 확인이 안될 가능성이 있기 때문.
crp_cd_list = crp_cd_df.values.flatten().tolist()
i = 0
# crp_cd = "005930"  # for test
ind_cd_list = []

# 안되면 여기부터 for루프까지 드래그 한 뒤 alt+shift+e로 다시 돌리면 되도록 코드를 안배.
last_checked_idx = i
print(len(ind_cd_list))
print(len(crp_cd_list))

for crp_cd in crp_cd_list[last_checked_idx:]:
    time.sleep(0.5)  # 너무 빨리하면 도중에 api가 응답을 안하는 경우가 있다.
    print(i)
    url = url_company.format(apikey, crp_cd)
    response = requests.get(url)
    print('done')
    data = json.loads(response.content.decode('utf-8'))
    # pprint(data)
    ind_cd_list.append(data['ind_cd'])
    i += 1

crp_ind_match_df = pd.DataFrame({'crp_cd': crp_cd_list, 'ind_cd': ind_cd_list})
crp_ind_match_df.to_excel('crp_ind_match.xlsx', index=False)  # 2061 row
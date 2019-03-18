
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
crpcode = "00126380"
crpcode = "005930"
url = url_company.format(apikey, crpcode)

response = requests.get(url)
data = json.loads(response.content.decode('utf-8'))
pprint(data)

url_search = "http://dart.fss.or.kr/api/search.json?auth={0}&crp_cd={1}&start_dt=20130101&bsn_tp=A001&bsn_tp=A002&bsn_tp=A003"
url = url_search.format(apikey, crpcode)
response = requests.get(url)
data = json.loads(response.content.decode('utf-8'))
pprint(data)

# 핵심 제무재표를 얻기 위한 dcm번호 구할 방법
url = "http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20160516003174"
req = requests.get(url)
tree = lxml.html.fromstring(req.text)
onclick = tree.xpath('//*[@id="north"]/div[2]/ul/li[1]/a')[0].attrib['onclick']
pattern = re.compile("^openPdfDownload\('\d+',\s*'(\d+)'\)")
dcm = pattern.search(onclick).group(1)
print(dcm)

url_search = "http://dart.fss.or.kr/api/search.json?auth={0}" \
             "&crp_cd={1}" \
             "&start_dt=20130101&end_dt=20181231" \
             "&bsn_tp=A001&bsn_tp=A002&bsn_tp=A003" \
             "&page_set=100"



# begin_date = '20110101'
begin_date = '20130101'
end_date = '20181231'
# end_date = '20131231'
crawl_start_time = datetime.now()
print("crawl_start_time : ", crawl_start_time)


def init():
    global driver
    driver = webdriver.Chrome()


def goto_bottom():
    old_len = 0
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(4)
        lis = driver.find_elements_by_xpath('//div[@class="stream"]/ol/li')
        if old_len == len(lis):
            break
        else:
            old_len = len(lis)


def goto_next_page():
    page_list = driver.find_element_by_class_name('page_list')
    current_page_num = int(page_list.find_elements_by_class_name('on')[0].text)
    max_page_num = int(page_list.find_elements_by_class_name('page_info')[0].text.split('/')[1].split(']')[0])
    if current_page_num != max_page_num:  # 마지막 페이지인지 확인
        current_page_num += 1
        driver.execute_script("search(" + str(current_page_num) + ")")  # 자바스크립트 함수 호출하는 방식 사용
        return wait_until_result_appear()
    else:  # 마지막 페이지라면
        print('any more page!')
        return False


def wait_until_result_appear():  # 로딩이 끝났는지 확인.
    delay = 7  # 로딩만 확인할 뿐 table_list가 나타난다고 검색결과가 0건이 아니란 보장은 없음. 100건 기준으로 일반적으로 7초면 로딩 완료.
    while True:
        try:
            element_present = EC.presence_of_element_located((By.CLASS_NAME, 'table_list'))
            WebDriverWait(driver, delay).until(element_present)
            # print("Page is ready!")
            return True
        except TimeoutException:
            print("Loading took too much time! ", delay, 'sec')


def check_page_existence():  # 검색결과가 없는지 확인. 있으면 True 없으면 False.
    page_list = driver.find_element_by_class_name('page_list')
    if len(page_list.find_elements_by_class_name('on')) != 0:  # 페이지가 하나라도 존재하면 검색결과가 존재.
        return True
    else:
        return False


def get_attached_footnote():  # 재무제표 본문의 주석을 스크래핑 위한 코드.
    doc_contents = driver.find_element_by_class_name('x-tree-root-node').find_elements_by_tag_name('li')  # 구조상 ul이 포함된 li까지 검출된다.
    for doc_content in doc_contents:  # 분기재무제표 ul 아래 있는 li에 주석이 있다면, ul, li 텍스트 둘다 나오게 된다. 그러나 최종적으로는 말단에 위치한 텍스트를 붙여 오게 된다.
        # 비효율적인 구간이므로 추후 개량을 요함.
        if "주석" in doc_content.text and doc_content.find_elements_by_tag_name('ul').text == "":
            print('at ', doc_content.text)
            doc_content.click()
            driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))
            foot_note = driver.find_element_by_xpath('/html/body').text
            driver.switch_to.default_content()
    return foot_note


def get_account_corporation(rcp_no):  # 잠정적으로 사용 중지.
    doc_contents = driver.find_element_by_class_name('x-tree-root-node').find_elements_by_tag_name('li')
    for doc_content in doc_contents:  # get_attached_footnote()와 마찬가지로 비효율적인 구간이므로 추후 개량을 요함.
        if doc_content.text.replace(" ", "") in ('감사보고서', '검토보고서'):
            print('at ', doc_content.text)
            doc_content.click()
            driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))
            text = driver.find_element_by_xpath('/html/body').text.strip()
            # for line in (line for line in text if line.rstrip('\n')):
            #     last = line
            last = text.splitlines()[-1]
            driver.switch_to.default_content()  # 일반적으로 보고서 명 맨 마지막에는 회계법인 글자가 들어간다.
    acc_crp = last.strip().replace(" ", "")
    if '법인' in acc_crp:
        return acc_crp
    else:
        print(acc_crp, " Fail to get account corporation name. ", rcp_no)
        return ""


def get_acc_crp(rcp_no):
    doc_contents = driver.find_element_by_class_name('x-tree-root-node').find_elements_by_tag_name('li')
    for doc_content in doc_contents:  # get_attached_footnote()와 마찬가지로 비효율적인 구간이므로 추후 개량을 요함.
        if doc_content.text.replace(" ", "") in ('감사보고서', '검토보고서'):
            print('at ', doc_content.text)
            doc_content.click()
            driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))
            text = driver.find_element_by_xpath('/html/body').text.strip()
            # for line in (line for line in text if line.rstrip('\n')):
            #     last = line
            last = text.splitlines()[-1]
            driver.switch_to.default_content()
    acc_crp = last.strip().replace(" ", "")
    if '법인' in acc_crp:
        return acc_crp
    else:
        print(acc_crp, " Fail to get account corporation name. ", rcp_no)
        return rcp_no


def make_sheet(crp_cls):
    df = pd.DataFrame()
    # tbl = driver.find_element_by_xpath("//*[@id='listContents']/div[1]/table").get_attribute('outerHTML')
    # tmp_df = pd.read_html(tbl) # 곧장 데이터 프레임으로 테이블을 변환시킬 수 있는 코드. 추후 주석의 테이블을 처리하는 식으로 쓸 수 있을것.
    # table_data = driver.find_elements_by_xpath('//*[@id="listContents"]/div[1]/table')
    tbody = driver.find_element_by_xpath('//*[@id="listContents"]/div[1]/table/tbody')
    i = 0
    # row = tbody.find_elements_by_tag_name('tr')[0] # for test
    for row in tbody.find_elements_by_tag_name('tr'):
        i += 1
        j = 0
        print('i', i)
        for cell in row.find_elements_by_tag_name('td'):
            # cell = row.find_elements_by_tag_name('td')[2]  # for test
            # print('j', j)
            if j == 0:
                j += 1
                continue
            elif j == 1:
                crp_nm = cell.text.strip()  # 공시대상회사(종목명)
                # cell.click()
                current_window = driver.current_window_handle
                # driver.execute_script("window.open('http://dart.fss.or.kr"+cell.find_element_by_tag_name('a').get_attribute('href')+"')")
                driver.execute_script("window.open('"+cell.find_element_by_tag_name('a').get_attribute('href')+"')")
                new_window = [window for window in driver.window_handles if window != current_window][0]
                driver.switch_to.window(new_window)

                crp_cd = driver.find_element_by_xpath('//*[@id="pop_body"]/div/table/tbody/tr[4]/td').text  # 종목코드

                # driver.execute_script("window.history.go(-1)")  # 뒤로 가기
                # driver.execute_script("closeCorpInfo(); return false;")  # 스크립트 실행
                driver.close()  # 현재 탭을 닫고 기존 검색결과 창으로 돌아감.
                driver.switch_to.window(current_window)
                j += 1
            elif j == 2:
                dic_cls = ''
                if '[' in cell.text:  # 만약 별도로 공시구분이 될만한 사항이 존재한다면.
                    dic_cls = cell.text.split(']')[0].split('[')[1]  # 공시구분. 아마 정기공시라면 기재정정과 그렇지 않은 두 종류 뿐일것.
                    rpt_nm = cell.text.split(']')[1]  # 보고서명. '공시구분+보고서명+기타정보'로 구성
                else:
                    rpt_nm = cell.text  # 보고서명. '공시구분+보고서명+기타정보'로 구성
                rcp_no = cell.find_element_by_tag_name('a').get_attribute('href').split('=')[1]  # 보고서 번호
                # cell.click()
                current_window = driver.current_window_handle
                driver.execute_script("window.open('http://dart.fss.or.kr/dsaf001/main.do?rcpNo="+rcp_no+"')")
                new_window = [window for window in driver.window_handles if window != current_window][0]
                driver.switch_to.window(new_window)
                doc_contents = driver.find_element_by_class_name('x-tree-root-node').find_elements_by_tag_name('li')
                # doc_contents = driver.find_element_by_class_name('x-tree-root-node').find_elements_by_xpath('//ul/li')

                foot_note = ''  # 재무제표 주석
                consolidated_foot_note = ''  # 연결재무제표 주석
                acc_crp = ''  # 감사 혹은 검토한 회계법인명

                # 재무제표 본문에서 있으면 주석 스크래핑. 없으면 감사인(법인명)이라도.
                for doc_content in doc_contents:
                    # print(doc_content.text)
                    if "연결재무제표 주석" in doc_content.text and 'ul' != doc_content.tag_name:  # 연결 재무제표가 그냥 계열사가 없다고만 서술 될 가능성도 존재.
                        print("연결재무제표 주석")
                        doc_content.click()
                        driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))  # iframe은 selenium으로 읽을 때 주의 사항. 버그 예상 지점.
                        consolidated_foot_note = driver.find_element_by_xpath('/html/body').text
                        driver.switch_to.default_content()
                        continue  # 연결재무제표 주석은 재무제표 주석이라는 문자열을 공통으로 포함하기 때문.
                    if "재무제표 주석" in doc_content.text and 'ul' != doc_content.tag_name:
                        print('재무제표 주석')
                        doc_content.click()
                        driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))
                        foot_note = driver.find_element_by_xpath('/html/body').text
                        driver.switch_to.default_content()
                    if "감사인" in doc_content.text:
                        print(doc_content.text)
                        doc_content.click()
                        driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))

                        texts = driver.find_element_by_xpath('/html/body').text.split('\n')

                        trigger_point = 0
                        for line in texts:
                            if '회계감사인의 명칭' in line:
                                trigger_point = 1
                                continue

                            if '법인' in line and trigger_point == 1:
                                for corporation in line.split():
                                    if '법인' in corporation:
                                        acc_crp = corporation  # 공시하는 회사명 법인명을 적었을 수 있다. 유난히 숫자가 적다면 확인이 요구됨.
                                break
                            # else:
                                # print('Account corporation not has word 법인 ', line)
                                # acc_crp = line
                        driver.switch_to.default_content()
                flag1 = 0
                flag2 = 0
                flag3 = 0
                flag4 = 0
                dcm_no = ''  # 감사보고서 번호
                col_dcm_no = ''  # 연결 감사 보고서 번호
                if foot_note == '':  # 만약 재무제표 상에서 주석을 긁어오지 못했다면 첨부파일을 찾을 것.
                    attach_doc_list = driver.find_element_by_id('att').find_elements_by_tag_name('option')
                    # select = Select(driver.find_element_by_id('att'))
                    for attach_doc in attach_doc_list:  # 모종의 이유로 로딩이 안되어서 선택 자체가 안되는 오류가 생김. 기다려도 안되는 것으로 보아 빠른 시간내에 해치우면 가망이 있다고 판단.
                        # print(attach_doc.text)
                        option_text = attach_doc.text.strip().split()
                        if len(option_text) > 1:  # 기타 다른 첨부 보고서명. # 형식이 '날짜 보고서명'을 따르기 때문에 띄워쓰기로 분할하면 2조각 이상 나와야 한다.
                            option_text = " ".join(option_text[1:])
                        else:  # +첨부선택+ 옵션만 여기서 걸러질 것.
                            if len(attach_doc_list) == 1:  # 첨부서류가 없는 경우
                                dcm_no = '0'
                                col_dcm_no = '0'
                                # acc_crp = 'unknown'  # 감사 받았다고는 써있는데 찾기 어려움. 보통 재무제표 본문 IV 감사인의 감사의견 등 에서 따로 찾아야한다.
                                print('no attachment!')
                                break
                            continue

                        # print(option_text)
                        if option_text in ('[정정] 감사보고서', '[정정] 반기검토보고서', '[정정] 분기검토보고서'):  # 정정인 경우.
                            dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            print('flag 1', dcm_no)
                            flag1 = 1
                            # driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + dcm_no)
                            # foot_note = get_attached_footnote()
                            # continue  # 한번만 회계법인명을 알아내면 되기 때문.
                        elif option_text in ('[정정] 연결감사보고서', '[정정] 반기연결검토보고서', '[정정] 분기연결검토보고서'):  # 정정인 경우.
                            col_dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            print('flag 2', col_dcm_no)
                            flag2 = 1
                            # driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + col_dcm_no)
                            # consolidated_foot_note = get_attached_footnote()
                            # continue
                        elif dcm_no == "" and option_text in ('감사보고서', '반기검토보고서', '분기검토보고서'):
                            print('flag 3', dcm_no)
                            dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            flag3 = 1
                            # driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + dcm_no)
                            # foot_note = get_attached_footnote()
                            # continue
                        elif col_dcm_no == "" and option_text in ('연결감사보고서', '반기연결검토보고서', '분기연결검토보고서'):
                            print('flag 4', col_dcm_no)
                            col_dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            flag4 = 1
                            # driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + col_dcm_no)
                            # consolidated_foot_note = get_attached_footnote()
                            # continue

                    if dcm_no != "":
                        driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + dcm_no)
                        foot_note = get_attached_footnote()
                    else:
                        dcm_no == "error "+rcp_no  # 만약 본문에도 주석을 못 얻었는데 이렇다면 문제가 있는것.

                    if col_dcm_no != "":
                        driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + col_dcm_no)
                        consolidated_foot_note = get_attached_footnote()
                    else:
                        col_dcm_no == "error "+rcp_no  # 만약 본문에도 주석을 못 얻었는데 이렇다면 문제가 있는것.

                    """
                    attach_doc_list = driver.find_element_by_id('att').find_elements_by_tag_name('option')
                    select = Select(driver.find_element_by_id('att'))
                    for attach_doc in attach_doc_list:
                        # print(attach_doc.text)
                        option_text = attach_doc.text.split()
                        if len(option_text) > 1:  # 기타 다른 첨부 보고서명. # 형식이 '날짜 보고서명'을 따르기 때문에 띄워쓰기로 분할하면 2조각 이상 나와야 한다.
                            option_text = " ".join(option_text[1:])
                        else:  # +첨부선택+ 옵션만 여기서 걸러질 것.
                            if len(attach_doc_list) == 1:  # 첨부서류가 없는 경우
                                dcm_no = '0'
                                col_dcm_no = '0'
                                acc_crp = 'unknown'  # 감사 받았다고는 써있는데 찾기 어려움. 보통 재무제표 본문 IV 감사인의 감사의견 등 에서 따로 찾아야한다.
                                break
                            continue
                        # print(i, '-', j, " : ", option_text)
                        if dic_cls != '' and option_text in ('[정정] 감사보고서', '[정정] 반기검토보고서', '[정정] 분기검토보고서'):
                            select.select_by_visible_text(attach_doc.text)
                            foot_note = get_attached_footnote()
                            dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            # print('flag 1', dcm_no)
                            acc_crp = get_account_corporation(rcp_no)
                            # print(dcm_no)
                            continue
                        if dic_cls != '' and option_text in ('[정정] 감사연결보고서', '[정정] 반기연결검토보고서', '[정정] 분기연결검토보고서'):  # 정정인 경우.
                            select.select_by_visible_text(attach_doc.text)
                            consolidated_foot_note = get_attached_footnote()
                            col_dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            # print('flag 2', col_dcm_no)
                            acc_crp = get_account_corporation(rcp_no)
                            continue
                        if dic_cls == '' and option_text in ('감사보고서', '반기검토보고서', '분기검토보고서'):
                            select.select_by_visible_text(attach_doc.text)
                            foot_note = get_attached_footnote()
                            dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            # print('flag 3', dcm_no)
                            acc_crp = get_account_corporation(rcp_no)
                            continue
                        if dic_cls == '' and option_text in ('연결감사보고서', '반기연결검토보고서', '분기연결검토보고서'):
                            select.select_by_visible_text(attach_doc.text)
                            consolidated_foot_note = get_attached_footnote()
                            col_dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            # print('flag 4', col_dcm_no)
                            acc_crp = get_account_corporation(rcp_no)
                            continue
                        """
                else:  # 이미 주석을 얻었다면 감사/검토 문서 번호 만 추출
                    attach_doc_list = driver.find_element_by_id('att').find_elements_by_tag_name('option')
                    for attach_doc in attach_doc_list:  # 모종의 이유로 로딩이 안되어서 선택 자체가 안되는 오류가 생김. 기다려도 안되는 것으로 보아 빠른 시간내에 해치우면 가망이 있다고 판단.
                        # print(attach_doc.text)
                        option_text = attach_doc.text.strip().split()
                        if len(option_text) > 1:  # 기타 다른 첨부 보고서명. # 형식이 '날짜 보고서명'을 따르기 때문에 띄워쓰기로 분할하면 2조각 이상 나와야 한다.
                            option_text = " ".join(option_text[1:])
                        else:  # +첨부선택+ 옵션만 여기서 걸러질 것.
                            if len(attach_doc_list) == 1:  # 첨부서류가 없는 경우
                                dcm_no = '0'
                                col_dcm_no = '0'
                                # acc_crp = 'unknown'  # 감사 받았다고는 써있는데 찾기 어려움. 보통 재무제표 본문 IV 감사인의 감사의견 등 에서 따로 찾아야한다.
                                print('no attachment!')
                                break
                            continue

                        # print(option_text)
                        if option_text in ('[정정] 감사보고서', '[정정] 반기검토보고서', '[정정] 분기검토보고서'):  # 정정인 경우.
                            dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            print('flag 1', dcm_no)
                            # driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + dcm_no)
                            # foot_note = get_attached_footnote()
                            # continue  # 한번만 회계법인명을 알아내면 되기 때문.
                        elif option_text in ('[정정] 연결감사보고서', '[정정] 반기연결검토보고서', '[정정] 분기연결검토보고서'):  # 정정인 경우.
                            col_dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            print('flag 2', col_dcm_no)
                            # driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + col_dcm_no)
                            # consolidated_foot_note = get_attached_footnote()
                            # continue
                        elif dcm_no == "" and option_text in ('감사보고서', '반기검토보고서', '분기검토보고서') and option_text != '감사의감사보고서':
                            print('flag 3', dcm_no)
                            dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            # driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + dcm_no)
                            # foot_note = get_attached_footnote()
                            # continue
                        elif col_dcm_no == "" and option_text in ('연결감사보고서', '반기연결검토보고서', '분기연결검토보고서'):
                            print('flag 4', col_dcm_no)
                            col_dcm_no = attach_doc.get_attribute('value').split('=')[2]
                            # driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no + '&dcmNo=' + col_dcm_no)
                            # consolidated_foot_note = get_attached_footnote()
                            # continue

                    if dcm_no == "":
                        dcm_no == '0'  # 이미 주석은 얻었으니 첨부파일이 없다면 없는것.
                    if col_dcm_no == "":
                        col_dcm_no == '0'  # 이미 주석은 얻었으니 첨부파일이 없다면 없는것.
                    """
                    for attach_doc in attach_doc_list:
                        print('wait')
                        ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
                        your_element = WebDriverWait(driver, 2, ignored_exceptions=ignored_exceptions) \
                            .until(expected_conditions.presence_of_element_located((By.ID, 'att')))
                        print(attach_doc.text)
                        # time.sleep(1)  # selector 옵션이 모두 로드 될때 까지 확인.
                        option_text = attach_doc.text.strip().split()
                        if len(option_text) > 1:  # 기타 다른 첨부 보고서명. # 형식이 '날짜 보고서명'을 따르기 때문에 띄워쓰기로 분할하면 2조각 이상 나와야 한다.
                            option_text = "".join(option_text[1:])
                        else:  # +첨부선택+ 옵션만 여기서 걸러질 것.
                            continue
                        print(i, '-', j, " : ", option_text)
                        if dic_cls != '' and option_text in ('[정정] 감사보고서', '[정정] 반기검토보고서', '[정정] 분기검토보고서'):
                            print('flag 1')
                            select.select_by_visible_text(attach_doc.text)
                            acc_crp = get_account_corporation(rcp_no)
                            continue
                        if dic_cls != '' and option_text in ('[정정] 감사연결보고서', '[정정] 반기연결검토보고서', '[정정] 분기연결검토보고서'):  # 정정인 경우.
                            print('flag 2')
                            select.select_by_visible_text(attach_doc.text)
                            acc_crp = get_account_corporation(rcp_no)
                            continue
                        if dic_cls == '' and option_text in ('감사보고서', '반기검토보고서', '분기검토보고서'):
                            print('flag 3')
                            select.select_by_visible_text(attach_doc.text)
                            acc_crp = get_account_corporation(rcp_no)
                            continue
                        if dic_cls == '' and option_text in ('연결감사보고서', '반기연결검토보고서', '분기연결검토보고서'):
                            print('flag 4')
                            select.select_by_visible_text(attach_doc.text)
                            acc_crp = get_account_corporation(rcp_no)
                            continue
                        """
                # print(foot_note)
                # print(consolidated_foot_note)

                if consolidated_foot_note == '3. 연결재무제표 주석\n- 해당사항 없음.':
                    print('연결재무제표 주석 해당사항 없음')
                    col_dcm_no = '0'
                    consolidated_foot_note == '0'  # 일단 내용이 없다는 것만이라도 표시를 해두는 것이 좋을 것.
                driver.close()  # 탭을 닫고 기존 검색결과 창으로 돌아감.
                driver.switch_to.window(current_window)
                j += 1
            elif j == 3:
                j += 1
                continue  # '제출인'은 딱히 감사보고서가 아닌이상 공시대상회사이다. 무의미하므로 제외.
            elif j == 4:
                rcp_dt = cell.text  # 접수일자. 형식은 'yyyy.mm.dd' 파싱에 주의.
                break  # 다음 칸은 생략. 비고의 '연' 과 같은 구분은 사실 별로 신뢰되지 않는다.(분기, 반기보고서는 연결제무제표인데도 '연'이 안붙은 경우가 있음) http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20181231000141

        # 보고서명에 있는 날짜(e.g. 2018년 3월). 현실적으로는 별로 중요치 않아 제외.
        # 법인등록번호 # 종목 코드가 있으므로 제외.
        # 공시가 올라온 당일부터 5 영업일~

        df.loc[i - 1, 'crp_cls'] = crp_cls  # 법인유형(유가증권Y, 코스닥K)
        df.loc[i - 1, 'crp_nm'] = crp_nm  # 공시대상회사(종목명)
        df.loc[i - 1, 'crp_cd'] = crp_cd  # 종목코드
        df.loc[i - 1, 'rpt_nm'] = rpt_nm  # 보고서명
        df.loc[i - 1, 'rcp_no'] = rcp_no  # 보고서 번호
        df.loc[i - 1, 'dic_cls'] = dic_cls  # 공시구분
        df.loc[i - 1, 'dcm_no'] = dcm_no  # 만약 주석이 다른 첨부파일에 있다면 그 첨부파일의 번호. 추후 재확인을 위함.
        df.loc[i - 1, 'col_dcm_no'] = col_dcm_no  # 만약 연결 주석이 다른 첨부파일에 있다면 그 첨부파일의 번호. 추후 재확인을 위함.
        df.loc[i - 1, 'foot_note'] = foot_note  # 주석
        df.loc[i - 1, 'consolidated_foot_note'] = consolidated_foot_note  # 연결재무제표 주석. 주석과 연결재무제표주석이 있다면 일단 연결재무제표 주석을 우선시해서 크롤링.
        df.loc[i - 1, 'acc_crp'] = acc_crp  # 감사보고서를 작성한 회사.
        df.loc[i - 1, 'rcp_dt'] = rcp_dt  # 접수일자

        # print('i', i)
    return df


def main():
    init()
    # crp_cls = input('법인유형(e.g. 유가증권시장이면 Y, 코스닥이면 K, 코넥스면 N, 기타는 E)를 대문자로 입력하세요.')
    crp_cls = 'Y'  # 테스트용
    now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')  # 파일이름 지을때 사용할 문자열(날짜형이 아님)
    driver.get('http://dart.fss.or.kr/dsab002/main.do')

    # 기간지정(코드상에 미리 지정)
    driver.find_element_by_id('startDate').clear()
    # driver.find_element_by_id('ext-gen81').send_keys(Keys.BACKSPACE)
    driver.find_element_by_id('ext-gen81').send_keys(Keys.RETURN)
    driver.find_element_by_name('startDate').send_keys(begin_date)

    driver.find_element_by_id('endDate').clear()
    driver.find_element_by_id('endDate').send_keys(end_date)

    # 최종보고서만 보는 것을 해제(실제로는 정정되지 않은 정보가 영향을 미쳤을 것이므로)
    driver.find_element_by_id('finalReport').click()

    # 업종지정 : 일단 업종 코드만 얻는 것이 목적이므로 생략. 추후 고려 요함.

    # 법인유형
    select = Select(driver.find_element_by_id('corporationType'))
    if crp_cls == 'Y':
        select.select_by_visible_text('유가증권시장')  # 참고로 value로 하려면 유가증권시장은 P. API와 다름.
    elif crp_cls == 'K':
        select.select_by_visible_text('코스닥시장')  # 참고로 value로 하려면 코스닥은 A. API와 다름.
    elif crp_cls == 'N':
        select.select_by_visible_text('코넥스시장')  # 참고로 value로 하려면 코넥스는 N. API와 동일.
    else:
        select.select_by_visible_text('기타법인')  # 참고로 value로 하려면 기타는 E. API와 동일.

    # 보고서 지정
    driver.find_element_by_id('publicTypeButton_01').click()  # 정기 공시
    driver.find_element_by_id('publicType1').click()  #
    driver.find_element_by_id('publicType2').click()
    driver.find_element_by_id('publicType3').click()

    # 최대 조회 수 지정(최대 100까지 볼 수 있다)
    select = Select(driver.find_element_by_id('maxResultsCb'))
    # select by visible text
    select.select_by_visible_text('100')

    driver.find_element_by_id('searchpng').click()
    wait_until_result_appear()
    check_more = check_page_existence()
    page_num = 0
    while check_more == True:
        # searchDateEnd = searchDateStart + timedelta(days=1)
        # searchDateStartStr = searchDateStart.strftime('%Y-%m-%d')
        # searchDateEndStr = searchDateEnd.strftime('%Y-%m-%d')
        page_num += 1
        print('page : ', page_num)
        df = make_sheet(crp_cls)  # 로드된 웹 요소들 긁어오기.
        df.to_excel(crp_cls + '_' + str(page_num) + '_crawlDate ' + now + '.xlsx', index=False)
        # check_more = goto_next_page()  # 다음 게시물 번호로 이동 일단 한페이지당 8분정도 소요되는 것으로 추정. 코스피 관련 결과를 모두 긁는다면
        break  # 테스트용.


# if __name__ == '__main__':
#     main()
main()
print("take time : {}".format(datetime.now() - crawl_start_time))
# driver.quit()

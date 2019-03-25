import pandas as pd
import os
from datetime import datetime

keyword = input('검색 및 저장할 키워드명을 입력하세요.')
print('wait sec')
# path_dir = input('엑셀 파일이 모인 곳의 경로를 복사 붙여넣기 하시오')
# path_dir = os.path.dirname(os.path.realpath(__file__))  # 현 스크립트가 있는 경로

# path_dir = 'C:/Users/jin/PycharmProjects/crawlDartFootNote/K_0_from20130101to20150817'  # done
# path_dir = 'C:/Users/jin/PycharmProjects/crawlDartFootNote/K_0_from20130101to20180515'  # done
# path_dir = 'C:/Users/jin/PycharmProjects/crawlDartFootNote/K_0_from20130101to20181231'  # done
# path_dir = 'C:/Users/jin/PycharmProjects/crawlDartFootNote/Y_0_from20130101to20141231'  # done
# path_dir = 'C:/Users/jin/PycharmProjects/crawlDartFootNote/Y_0_from20150101to20151231'  # done
# path_dir = 'C:/Users/jin/PycharmProjects/crawlDartFootNote/Y_0_from20160101to20181114'  # done
# path_dir = 'C:/Users/jin/PycharmProjects/crawlDartFootNote/Y_0_from20160101to20181231'  # done
path_dir = 'C:/Users/jin/PycharmProjects/crawlDartFootNote'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)

print(path_dir)

file_list = os.listdir(path_dir)
file_list.sort()
filtered_file_df = []

for path, dirs, files in os.walk(path_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() == '.xlsx':
            # print('excel 파일: ' + file)
            if keyword in file.split(".")[0]:
                # print(path+file)
                tmp_df = pd.read_excel(path + "\\" + file, dtype=str)  # dtype=str option이 없으면 종목코드가 숫자로 처리됨.
                if not tmp_df.empty:
                    filtered_file_df.append(tmp_df)

df = pd.concat(filtered_file_df, join='inner', axis=0)
print(df[df.duplicated(keep=False)])
# pd.concat([df1,df2]).drop_duplicates().reset_index(drop=True)
df = df.drop_duplicates()

## csv방식 : csv로 저장하면 쓰고 읽는데 오류는 안 생기지만 기가바이트 단위가 되어버린다.
# df.to_excel('total ' + path_dir.split('/')[-1] +' data collection.xlsx', index=False)
# df1 = pd.read_csv('total ' + path_dir.split('/')[-1] +' data collection.csv', dtype=str)
# df.equals(df1)

## pickle 방식 : 메모리 넘기지만 않으면 잘 작동하지만 여전히 용량이 기가바이트.
# print(df.dtypes)
# df2 = df.astype('string')
# print(df2.dtypes)
# df2.to_pickle('total ' + path_dir.split('/')[-1] +' data collection')
# df3 = pd.read_pickle('total ' + path_dir.split('/')[-1] +' data collection')
# print(df.equals(df3))

writer = pd.ExcelWriter('total ' + path_dir.split('/')[-1] +' data collection.xlsx', engine='xlsxwriter')  # import xlsxwriter 깔아야 사용 가능한 방법
df.to_excel(writer, 'Sheet1', index=False)
writer.save()
'''
* 데이터 품질 체크 결과 유의 할 사항
- 유가증권시장 AJ네트웍스는 2014.05.15에(즉 같은날) 두 번의 첨부정정이 존재한다. 하나는 연결감사보고서 첨부 서
류에, 하나는 감사보고서 첨부서류가 변경 되었기 때문. 본문에 주석이 없기 때문에 (연결)감사보고서 주석을 쓸 수 
밖에 없다. 알고리즘 상 최초에 감사보고서의 특수관계자 주석 기재 오류가 생겨도, 연결재무제표 주석은 해당 제출일
에서 가장 가까운 것으로 사용하기 때문에 벌어진 일로 추정된다. 일 단위로 분석할 것이므로 가장 마지막 결과만 써
도 결론적으론 큰 차이가 없다고 보고 들어간다. 일단 duplicates() 함수를 쓸 땐 keep='first'가 디폴트이기 때문에 
최종적으론 그날 바꾼 마지막 공시만 남게 될 것. 본문에 별도로 주석이 변한건 없으니 그대로 반영될것.
Y_24_from20130101to20141231_crawlDate_2019-03-17 20-26-56.xlsx 확인 결과

- 코스닥시장 미래에셋제5호스팩은 2018.03.29에(즉 같은날) 두번의 첨부정정이 존재한다. 때문에 중복 검증을 하면 
걸리는 데 이중에 감사보고서 주주 현황의 주석을 변경시킨 내역이 있다(본문도 동일). 본문 기준으로 크롤링 할 수 
있는 형식인데 본문은 바뀐 내역이 별도로 기록되지 않고 그냥 수정되어 있음. 정보 손실은 없다고 보면 된다. 일단 
duplicates() 함수를 쓸 땐 keep='first'가 디폴트이기 때문에 최종적으로 그날 바꾼 마지막 공시만 남게 될 것.
K_25_from20130101to20180515_crawlDate_2019-03-18 17-40-50.xlsx 확인 결과

- 유가증권시장 JW홀딩스 2016.03.30에(즉 같은날) 두 번의 첨부정정이 존재한다. 하나는 연결감사보고서 첨부 서류에
, 하나는 감사보고서 첨부서류가 변경 되었기 때문. 본문 기준으로 크롤링 수 있는 형식인데 본문은 바뀐 내역이 별도
로 기록되지 않고 그냥 수정되어 있음. 일단 duplicates() 함수를 쓸 땐 keep='first'가 디폴트이기 때문에 최종적으
론 그날 바꾼 마지막 공시만 남게 될 것. 본문에 별도로 주석이 변한건 없으니 그대로 반영될것.
Y_94_from20160101to20181114_crawlDate_2019-03-16 21-12-36.xlsx
'''

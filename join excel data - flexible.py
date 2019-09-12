import pandas as pd
import os
# dictionary 만들기 위해 total을 모으는 용도.
keyword = input('검색 및 저장할 키워드명을 입력하세요.')

path_dir = input("엑셀 파일이 모인 곳의 경로를 복사 붙여넣기 하시오. \nC:/Users/jin/PycharmProjects/crawlDartFootNote/financial ratio for dependent variable retrived 2019_05_15")
# path_dir = os.path.dirname(os.path.realpath(__file__))
file_list = os.listdir(path_dir)
file_list.sort()
filtered_file_df = []

for path, dirs, files in os.walk(path_dir):
    for file in files:
        if os.path.splitext(file)[1].lower() == '.xlsx':
            # print('csv 파일: ' + file)
            if keyword in file.split(".")[0]:
                print(file)
                tmp_df = pd.read_excel(path + "\\" + file, dtype=object)
                print(tmp_df.shape)
                if not tmp_df.empty:
                    filtered_file_df.append(tmp_df)

df = pd.concat(filtered_file_df, axis=1, join_axes=[filtered_file_df[0].index])
df = df.loc[:, ~df.columns.duplicated()]  # 중복되는 열 제거. 이 symbol 그런거.

# df = df.drop_duplicates()
# if 'dramaTitle' in list(df):
#     df = df.drop(columns=['dramaTitle', 'rt_user', 'sourceLink'])
# df.to_csv('total '+keyword+' data collection.csv', index=False, encoding='utf-8')
df.to_excel('total '+keyword+' data collection.xlsx', index=False, encoding='utf-8')

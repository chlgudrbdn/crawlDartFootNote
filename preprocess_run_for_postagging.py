"""
해당 스크립트는 postagging을 위한 코드. 별도로 자주 실행할 일이 많다보니 분리하여 작성.
"""
import preprocess_footnotes_data as pfd
import pandas as pd
import join_pickle_data as jpd
from datetime import datetime, timedelta
import numpy as np
from scipy import sparse
from konlpy.tag import Kkma
from konlpy.tag import Komoran
from konlpy.tag import Okt
from konlpy.tag import Hannanum
import gc
import re


def xplit(*delimiters):
    return lambda value: re.split('|'.join([re.escape(delimiter) for delimiter in delimiters]), value)


def divide_by_specific_number(df, max_rows):  # dataframe이 너무 큰 경우 자르기 위함. 리스트 형태로 넘겨주는 만큼 헛짓거리일 수도 있지만.
    list_of_dfs = []
    while len(df) > max_rows:
        top = df[:max_rows]
        # print(top.shape)
        list_of_dfs.append(top)
        df = df[max_rows:]
        # print(df.shape)
    else:
        list_of_dfs.append(df)
    return list_of_dfs


def foot_note_pos_tagging(df6, morpheme_ref, how_to_pos_treat_as_feature):  # df5 받아서 형태소 분석. df6라 명명했는데, 효율을 위함.
    for index, row in df6.iterrows():
        if index % 100 == 0:
            print(index)  # 단순 진척도 보기 위한 것.
        # print(index, row)  # for test
        row['foot_note'] = re.sub('\n+', "\n", row['foot_note'])  # 여러번 줄바꿈은 하나로
        row['foot_note'] = re.sub('\n ', "\n", row['foot_note'])  # 줄바꿈 다음 띄워쓰기 한칸은 그냥 띄워쓰기로
        row['foot_note'] = re.sub(' {2,}', "", row['foot_note'])  # 구   분 같이 표에 명시된 경우가 있는데, 서로 다른 두글자로 인식되는 경우를 제거하기 위함.
        # before_text = "   "  # for test
        # after_text = before_text  # for test
        # after_text = re.sub('[/[\{\}\[\]\/?|\)*~`!\-_+<>@\#$%&\\\=\(\'\"]+', '0', before_text)  # for test
        # after_text = re.sub(' {2,}', '0', before_text)  # for test
        sentences = xplit('. ', '? ', '! ', '\n', '.\n')(row['foot_note'])  # 문장별 구분
        morphemes = []
        i = 0
        for sentence in sentences:  # 이 절차를 미리하는 것은 사실 비효율이지만 에러 처리를 위해 임시방편으로 사용. 그리고 이걸 써야 한나눔의 글자수 제한도 피해갈 수 있다.
            # print(sentence)  # for test
            try:
                if how_to_pos_treat_as_feature == 'use_only_morph':  # 그냥 형태소만 적당히 잘라서 사용.
                    morphemes.extend(morpheme_ref.morphs(sentence))
                elif how_to_pos_treat_as_feature == 'attach_tag_to_pos':  # 한쌍의 튜플로 이뤄진 리스트를 품사까지 하나의 feature로 취급
                    # print('pos tagging start')  # for test
                    pos_tag_pair_list = morpheme_ref.pos(sentence)
                    # print(pos_tag_pair_list)  # for test
                    tmp = ''
                    for word in pos_tag_pair_list:
                        # tmp.append(("/".join(word))+';')
                        tmp = tmp + (("/".join(word))+';')
                    # print('tag result: ', tmp)  # for test
                    morphemes.extend(tmp)
                    del tmp
                    del pos_tag_pair_list
                elif how_to_pos_treat_as_feature == 'seperate_tag_and_pos':  # 추가적으로 이용을 위해 형태소 분류까지도 별도의 리스트의 섹터로 넣어서 취급.
                    morphemes.extend(morpheme_ref.pos(sentence))
            except Exception:
                continue
                print('error index', index, ' ', i, 'th sentence', sentence)
            i += 1
        # df6.at[index, 'foot_note'] = morphemes
        df6.at[index, 'foot_note'] = ''.join(morphemes)
        del morphemes
        del sentences
        # print(df6.at[index, 'foot_note'])
        # df6.loc[index, 'foot_note'] = morphemes  # bug
        # df6.loc[index].at['morpheme'] = morphemes  # bug
        # pos_tagged_lists.append(morphemes)  # bug
        # df6['morpheme'] = pos_tagged_lists  # bug
    # print(df6['foot_note'])
    # print(df6.info())
    print('done')
    return df6


def test_pos_for_check(df6, morpheme, how_to_pos_treat_as_feature):  # 형태소 붙이기. 직접적으로 붙이는건 아니고 설정을 관할하는 함수.
    start_time = datetime.now()
    print("start_time : ", start_time)
    if morpheme == 'kkma':
        morpheme_ref = Kkma()  # 꼬꼬마  # 오래걸린다. 제외.
    elif morpheme == 'komoran':
        morpheme_ref = Komoran()  # 코모란  # 성능은 좋은데 너무 좋다.
    elif morpheme == 'okt':
        morpheme_ref = Okt()  # 오픈 코리아 텍스트
    elif morpheme == 'hannanum':
        morpheme_ref = Hannanum()  # 한나눔  # 띄어쓰기 처리수준이 낮긴한데 이점이 오히려 합성어가 많은 재무 도메인의 경우 의미가 있을 수 있어 보인다.  # 라고 생각했는데 오류 투성이다.
    columns = ['rcp_no', 'dic_cls', 'dcm_no', 'col_dcm_no', 'consolidated_foot_note']
    df6.drop(columns, inplace=True, axis=1)

    directory_name = './divide_by_sector'
    # filename = 'filter6 '+morpheme+'_'+how_to_pos_treat_as_feature+'.pkl'
    df_list = divide_by_specific_number(df6, 5000)
    del df6
    # df_list_sector, sector_list = divide_by_sector(df6, filename, directory_name)
    # for tmp_df, sector in zip(df_list_sector, sector_list):
    print('divide done')
    len_of_df_list = len(df_list)
    for i in range(len_of_df_list):
        tmp_df = df_list[0]  # 앞에서 하나씩 지우는 pop 같은 방식
        tmp_df = foot_note_pos_tagging(tmp_df, morpheme_ref, how_to_pos_treat_as_feature)
        tmp_df.to_pickle(directory_name + '/filter6 '+morpheme+'_'+how_to_pos_treat_as_feature+'_'+str(i)+'.pkl')
        del tmp_df
        del df_list[0]  # 앞에서 하나씩 지우는 pop 같은 방식
        print("take time : {}".format(datetime.now() - start_time))
        i += 1
        gc.collect()
        # 2번것 다 끝나면
        # 파일 불러올땐 morpheme, how_to_pos_treat_as_feature로 걸러서 출력.
    print("end time : {}".format(datetime.now()))


df = pd.read_pickle('filter5_4 alternate col to footnotes.pkl')
# 이 단계에서 카테고리로 바꾸는 식으로 데이터 프레임의 크기를 좀 줄여야 한다.

df = df.reset_index(drop=True)
# morpheme = 'kkma'
# morpheme = 'komoran'
# morpheme = 'okt'
morpheme = 'hannanum'

# how_to_pos_treat_as_feature = 'use_only_morph'
how_to_pos_treat_as_feature = 'attach_tag_to_pos'
# how_to_pos_treat_as_feature = 'seperate_tag_and_pos'

test_pos_for_check(df, morpheme, how_to_pos_treat_as_feature)  # divide_by_sector에 알아서 파일로 저장. 별도로 return은 불필요.

path_dir = 'C:/Users/lab515/PycharmProjects/crawlDartFootNote/divide_by_sector'
df = jpd.join_pickle_data(path_dir, morpheme)
df.to_pickle('filter6 hannanum_attach_tag_to_pos.pkl')

# jpd.join_pickle_data()
# fnguide_fin_ratio_dat = pd.read_excel('main_dependant_var.xlsx', dtype=object)
# file_name = 'filter6 hannanum_attach_tag_to_pos_0.pkl'
# df = pd.read_pickle('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'+file_name)
# file_name = 'filter6 komoran_attach_tag_to_pos_0.pkl'
# df1 = pd.read_pickle('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'+file_name)
# df = pfd.merge_fnguide_data_and_filter_no_data(df, fnguide_fin_ratio_dat, file_name)
